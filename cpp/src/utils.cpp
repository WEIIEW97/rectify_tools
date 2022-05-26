#include "utils.h"

Eigen::MatrixXd normc(Eigen::MatrixXd x) {
    Eigen::MatrixXd _norm(x.rows(), x.cols());
    if (x.sum() / x.size() >= 0) {
        for (int i = 0; i < x.cols(); i++) {
            _norm.col(i) = x.col(i) / x.col(i).maxCoeff();
        }
    } else {
        for (int j = 0; j < x.rows(); j++) {
            for (int k = 0; k < x.cols(); k++) {
                _norm(j, k) = (x(j, k) - x.col(j).minCoeff()) /
                              (x.col(j).maxCoeff() - x.col(j).minCoeff());
            }
        }
    }
    return _norm;
}

void meshgrid(const cv::Range& xgv, const cv::Range& ygv, cv::Mat& X,
              cv::Mat& Y) {
    std::vector<int> t_x, t_y;
    for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
    for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);

    // need to transpose X
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}

// dec2bin inplementation
std::string dec2bin(int in_num, int bits_num) {
    std::string bin_str = "";
    int i = 0;
    while (in_num != 0) {
        bin_str = (in_num % 2 == 0 ? "0" : "1") + bin_str;
        in_num /= 2;
        i++;
    }
    while (i < bits_num) {
        bin_str = "0" + bin_str;
        i++;
    }
    return bin_str;
}

// convert binary string to decimal
int bin2dec(std::string bin_str) {
    int dec_num = 0;
    for (int i = 0; i < bin_str.length(); i++) {
        dec_num += (bin_str[i] - '0') * pow(2, bin_str.length() - i - 1);
    }
    return dec_num;
}

int32_t double2fixed(double num, int frac_len) {
    int32_t res;
    int _buffer;
    int int_part = 2 << (frac_len - 1);

    if (num > 0) {
        // after tons of experiments, should use flooring when input is positive
        // use ceiling when input is negative.
        _buffer = std::floor(num * int_part);
        res =
            (_buffer == int32_t(_buffer)) ? _buffer : (_buffer >> 31) ^ SIGN_BIT_IGNORE;
    } else if (num < 0) {
        num = -num;
        _buffer = std::ceil(num * int_part);
        res =
            (_buffer == int32_t(_buffer)) ? _buffer : (_buffer >> 31) ^ SIGN_BIT_IGNORE;
        res = res ^ SIGN_BIT;
    } else {
        res = 0;
    }
    return res;
}

double fixed2double(int32_t num, int frac_len) {
    int sign_flag = num & SIGN_BIT;  // 1 -> negative, 0 -> positive
    double res, _buffer;
    int32_t _buffer1;
    int int_part = 2 << (frac_len - 1);

    if (sign_flag == 0) {
        res = double(num) / int_part;
    } else {
        _buffer1 = num ^ SIGN_BIT;
        _buffer = double(_buffer1);
        res = _buffer / int_part;
        res = -res;
    }
    return res;
}

int sub2ind(int w, int h, int rows, int cols) {
    int ind = rows + cols * w;
    return ind;
}

int sub2ind_along_y(int w, int h, int rows, int cols) {
    int ind = cols + rows * h;
    return ind;
}

/*
usage: auto [w, h] = in2sub(...)
*/
std::tuple<int, int> ind2sub(int w, int h, int ind) {
    int rows = ind % h;
    int cols = ind / h;
    return std::make_tuple(rows, cols);
}

std::tuple<int, int> ind2sub_along_y(int w, int h, int ind) {
    int rows = ind / w;
    int cols = ind % w;
    return std::make_tuple(rows, cols);
}

// check an element if it is in a vector
bool ismember(double num, std::vector<double> vec) {
    if (std::binary_search(vec.begin(), vec.end(), num)) {
        return true;
    } else {
        return false;
    }
}

// extract y, u, v channels respectivily from a 3-channel image
void get_yuv(const std::string& yuv_file, int width, int height, cv::Mat& y,
             cv::Mat& u, cv::Mat& v) {
    std::ifstream file(yuv_file, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Cannot open file: " << yuv_file << std::endl;
        exit(1);
    }
    y.create(height, width, CV_8UC1);
    u.create(height / 2, width / 2, CV_8UC1);
    v.create(height / 2, width / 2, CV_8UC1);
    file.read((char*)y.data, y.total() * y.elemSize());
    file.read((char*)u.data, u.total() * u.elemSize());
    file.read((char*)v.data, v.total() * v.elemSize());
    file.close();
}

// get opencv mat type name
std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:
            r = "8U";
            break;
        case CV_8S:
            r = "8S";
            break;
        case CV_16U:
            r = "16U";
            break;
        case CV_16S:
            r = "16S";
            break;
        case CV_32S:
            r = "32S";
            break;
        case CV_32F:
            r = "32F";
            break;
        case CV_64F:
            r = "64F";
            break;
        default:
            r = "User";
            break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

// write cv::Mat to .csv
void write_csv(std::string file, cv::Mat m) {
    std::ofstream f;
    f.open(file.c_str());
    f << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
    f.close();
}