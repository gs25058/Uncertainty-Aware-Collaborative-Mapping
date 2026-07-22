/**
 * mono_miluv.cc  --  ORB-SLAM3 monocular front-end driver for the MILUV dataset.
 *
 * Reads a plain-text list file whose lines are "<timestamp_seconds> <image_path>"
 * (sorted by timestamp), feeds each frame to ORB-SLAM3 monocular, and writes the
 * keyframe trajectory (TUM format: t tx ty tz qx qy qz qw) used by the CoVOR-SLAM
 * fusion stage.
 *
 * Usage:
 *   ./mono_miluv path_to_vocabulary path_to_settings path_to_list_file output_traj.txt
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <System.h>

using namespace std;

static void LoadList(const string &listFile,
                     vector<string> &imgs, vector<double> &ts)
{
    ifstream f(listFile.c_str());
    if(!f.is_open()) { cerr << "Cannot open list file: " << listFile << endl; exit(1); }
    string line;
    while(getline(f, line))
    {
        if(line.empty()) continue;
        stringstream ss(line);
        double t; string path;
        ss >> t >> path;
        if(path.empty()) continue;
        ts.push_back(t);
        imgs.push_back(path);
    }
}

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << "Usage: ./mono_miluv vocab settings list_file output_traj.txt" << endl;
        return 1;
    }
    const string vocab    = argv[1];
    const string settings = argv[2];
    const string listFile = argv[3];
    const string outTraj  = argv[4];

    vector<string> vImg; vector<double> vTs;
    LoadList(listFile, vImg, vTs);
    const int N = (int)vImg.size();
    cout << "MILUV mono: " << N << " frames from " << listFile << endl;
    if(N == 0) return 1;

    // Headless: viewer disabled (4th arg = false).
    ORB_SLAM3::System SLAM(vocab, settings, ORB_SLAM3::System::MONOCULAR, false);
    const float imageScale = SLAM.GetImageScale();

    for(int ni = 0; ni < N; ni++)
    {
        cv::Mat im = cv::imread(vImg[ni], cv::IMREAD_GRAYSCALE);
        if(im.empty()) { cerr << "Failed to load: " << vImg[ni] << endl; return 1; }

        if(imageScale != 1.f)
        {
            int w = im.cols * imageScale, h = im.rows * imageScale;
            cv::resize(im, im, cv::Size(w, h));
        }

        SLAM.TrackMonocular(im, vTs[ni]);

        if((ni % 500) == 0)
            cout << "  processed " << ni << "/" << N << " (t=" << vTs[ni] << ")" << endl;
    }

    SLAM.Shutdown();

    // Keyframe trajectory in TUM format: timestamp tx ty tz qx qy qz qw
    SLAM.SaveKeyFrameTrajectoryTUM(outTraj);
    cout << "Saved keyframe trajectory -> " << outTraj << endl;

    return 0;
}
