#include <vector>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <xnn.h>
#include <picpac-cv.h>

using namespace std;

int main () {
    int mode = 1;
    xnn::Model::set_mode(mode);
    picpac::ImageStream::Config config;
    config.loop = false;
    config.shuffle = true;
    config.stratify = true;
    config.split=10;
    config.split_fold=0;
    config.split_negate=true;
    picpac::ImageStream db("db", config);

    xnn::Model *model = xnn::Model::create("model");
    CHECK(model);
    double loss = 0;
    int cnt = 0;
    for (;;) {
        try {
            picpac::ImageStream::Value v(db.next());
            vector<float> prob;
            model->apply(v.image, &prob);
            loss += std::log(prob[v.label]);
            cnt += 1;
            //cout << a << endl;
        }
        catch (picpac::EoS const &) {
            break;
        }
    }
    cout << -loss/cnt << endl;
    delete model;
    return 0;
}
