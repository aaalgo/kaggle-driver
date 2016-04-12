#include <vector>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <xnn.h>

using namespace std;

int main () {
    int mode = 1;
    xnn::Model::set_mode(mode);
    ifstream list("list");
    string path;
    string prefix = "test/";
    cout << "img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9" << endl;
    xnn::Model *model = xnn::Model::create("model");
    CHECK(model);
    while (list >> path) {
        cv::Mat image(cv::imread(prefix + path));
        vector<float> prob;
        model->apply(image, &prob);
        CHECK(prob.size() == 10);
        cout << path;
        for (auto &v: prob) {
            cout << ',' << v;
        }
        cout << endl;
    }
    delete model;
    return 0;
}
