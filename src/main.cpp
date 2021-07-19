///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/*********************************************************************************
 ** This sample demonstrates how to capture 3D point cloud and detected objects  **
 **      with the ZED SDK and display the result in an OpenGL window. 	        **
 **********************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

Camera zed;
bool exit_=false;
bool newFrame=false;


void parseArgs(int argc, char **argv,sl::InitParameters& param)
{
    if (argc > 1 && string(argv[1]).find(".svo")!=string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        cout << "[Sample] Using SVO File input: " << argv[1] << endl;
    } else if (argc > 1 && string(argv[1]).find(".svo")==string::npos) {
        string arg = string(argv[1]);
        unsigned int a,b,c,d,port;
        if (sscanf(arg.c_str(),"%u.%u.%u.%u:%d", &a, &b, &c, &d,&port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a)+"."+to_string(b)+"."+to_string(c)+"."+to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()),port);
            cout<<"[Sample] Using Stream input, IP : "<<ip_adress<<", port : "<<port<<endl;
        }
        else  if (sscanf(arg.c_str(),"%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            cout<<"[Sample] Using Stream input, IP : "<<argv[1]<<endl;
        }
        else if (arg.find("HD2K")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD2K;
            cout<<"[Sample] Using Camera in resolution HD2K"<<endl;
        } else if (arg.find("HD1080")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD1080;
            cout<<"[Sample] Using Camera in resolution HD1080"<<endl;
        } else if (arg.find("HD720")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD720;
            cout<<"[Sample] Using Camera in resolution HD720"<<endl;
        } else if (arg.find("VGA")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::VGA;
            cout<<"[Sample] Using Camera in resolution VGA"<<endl;
        }
    } else {
        //
    }
}

void run() {

    RuntimeParameters rtp;
    rtp.sensing_mode = SENSING_MODE::FILL;
    while(!exit_) {
        if (!newFrame) {
            if (zed.grab(rtp) == sl::ERROR_CODE::SUCCESS) {
                newFrame=true;
            }
        }
        sl::sleep_us(100);
    }
}

int main(int argc, char **argv) {
    // Create ZED objects
    InitParameters initParameters;
    initParameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    initParameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    initParameters.coordinate_units = sl::UNIT::METER;
    initParameters.depth_maximum_distance =30.f; //For object detection, Objects after 15meters may not be precise enough.
    parseArgs(argc,argv, initParameters);

    // Open the camera
    std::cout<<" Opening Camera"<<std::endl;
    ERROR_CODE zed_error = zed.open(initParameters);
    if (zed_error != ERROR_CODE::SUCCESS) {
        std::cout << sl::toVerbose(zed_error) << "\nExit program." << std::endl;
        zed.close();
        return 1; // Quit if an error occurred
    }

    Resolution resolution = zed.getCameraInformation().camera_resolution;
    auto camera_parameters = zed.getCameraInformation(resolution).calibration_parameters.left_cam;

    //Only ZED2 and ZED2i have object detection
    sl::MODEL model = zed.getCameraInformation().camera_model;
    if (model==sl::MODEL::ZED || model==sl::MODEL::ZED_M) {
        std::cout<<" ERROR : Use ZED2/ZED2i Camera only"<<std::endl;
        exit(0);
    }

    // Enable Position tracking (mandatory for object detection)
    std::cout<<" Enable Positional Tracking "<<std::endl;
    zed_error =  zed.enablePositionalTracking();
    if (zed_error != ERROR_CODE::SUCCESS) {
        std::cout << sl::toVerbose(zed_error) << "\nExit program." << std::endl;
        zed.close();
        return 1;
    }

    // Enable the Objects detection module
    std::cout<<" Enable Object Detection Module"<<std::endl;
    sl::ObjectDetectionParameters obj_det_params;
    obj_det_params.image_sync = true;
    zed_error = zed.enableObjectDetection(obj_det_params);
    if (zed_error != ERROR_CODE::SUCCESS) {
        std::cout << sl::toVerbose(zed_error) << "\nExit program." << std::endl;
        zed.close();
        return 1;
    }

    // Create OpenGL Viewer
    GLViewer viewer;
    viewer.init(argc, argv, camera_parameters);

    // Object Detection runtime parameters
    float object_confidence = 40.f;
    ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    objectTracker_parameters_rt.detection_confidence_threshold = object_confidence;
    objectTracker_parameters_rt.object_class_filter.clear();
    objectTracker_parameters_rt.object_class_filter.push_back(sl::OBJECT_CLASS::PERSON);

    // Create ZED Objects
    Objects objects;
    Mat pDepth,pImage;
    sl::Timestamp current_im_ts;

    // Capture Thread (grab will run in the thread)
    exit_=false;
    std::thread runner(run);

    // Update 3D loop
    while (viewer.isAvailable()) {
        if (newFrame) {
            //Retrieve Images and Z-Buffer
            zed.retrieveMeasure(pDepth, MEASURE::DEPTH, MEM::GPU);
            zed.retrieveImage(pImage, VIEW::LEFT, MEM::GPU);
            //Retrieve Objects
            zed.retrieveObjects(objects, objectTracker_parameters_rt);
            current_im_ts = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);
            newFrame=false;
            //Update GL view
            viewer.updateData(pImage,pDepth, objects,current_im_ts);


            //std::cout<<" NUmero Frame : "<<zed.getSVOPosition()<<std::endl;
        }
        else
            sleep_ms(1);
    }

    // OUT
    exit_=true;
    runner.join();
    pImage.free();
    pDepth.free();
    objects.object_list.clear();

    // Disable modules
    zed.disablePositionalTracking();
    zed.disableObjectDetection();
    zed.close();
    return 0;
}
