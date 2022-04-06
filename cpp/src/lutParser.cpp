#include "utils.h"


// read lookup table file and input image file
void lutParser(std::string lut_file, std::string input_image_file, int int_len, int frac_len) {
    cv::Mat input_img = cv::imread(lut_file);

    cv::Mat rect_img;
    int num, cnt = 0;
    std::vector<double> lut_info;
    FILE *fp = fopen(lut_file.c_str(), "r");
    if (fp == NULL) {
        printf("open file error!\n");
        return;
    }

    while (!std::feof(fp)) {
        char line[1024];
        num = std::stoi(fgets(line, 1024, fp));
        cnt++;
        lut_info.push_back(num);
    }

    fclose(fp);
    int index = 1;
    int col_num, row_num, max_diff;
    col_num = lut_info[index++];
    row_num = lut_info[index++];
    if (row_num > col_num) max_diff = row_num;
    else max_diff = col_num;

    int rect2raw_sample_row_num, rect2raw_sample_col_num, raw2rect_sample_row_num, raw2rect_sample_col_num;
    rect2raw_sample_row_num = lut_info[index++];
    rect2raw_sample_col_num = lut_info[index++];
    raw2rect_sample_row_num = lut_info[index++];
    raw2rect_sample_col_num = lut_info[index++];

    index = 14;
    std::vector<double> raw2rectSampleX, raw2rectSampleY, rect2rawSampleX, rect2rawSampleY;
    // generate raw2rectsampleX from index to index + raw2rect_sample+col_num - 1
    for (int i = 0; i < raw2rect_sample_col_num; i++) {
        raw2rectSampleX.push_back(lut_info[index++] / 2);
    }
    index += raw2rect_sample_col_num;

    // generate raw2rectsampleY from index to index + raw2rect_sample+row_num - 1  
    for (int i = 0; i < raw2rect_sample_row_num; i++) {
        raw2rectSampleY.push_back(lut_info[index++] / 2);
    }
    index += raw2rect_sample_row_num;

    // find index raw2rectSampleX > 2*max_diff
    for (int i = 0; i < raw2rectSampleX.size(); i++) {
        if (raw2rectSampleX[i] > 2 * max_diff) {
            raw2rectSampleX[i] = raw2rectSampleX[i] - pow(2, 14);
        }
    }

    // do the same for raw2rectSampleY
    for (int i = 0; i < raw2rectSampleY.size(); i++) {
        if (raw2rectSampleY[i] > 2 * max_diff) {
            raw2rectSampleY[i] = raw2rectSampleY[i] - pow(2, 13);
        }
    }

    // if raw2rectSampleX < -max_diff, then make it to raw2rectSampleX + 2^14
    for (int i = 0; i < raw2rectSampleX.size(); i++) {
        if (raw2rectSampleX[i] < -max_diff) {
            raw2rectSampleX[i] = raw2rectSampleX[i] + pow(2, 13);
        }
    }

    // generate rect2rawSampleX from index to index + rect2raw_sample+col_num - 1
    for (int i = 0; i < rect2raw_sample_col_num; i++) {
        rect2rawSampleX.push_back(lut_info[index++] / 2);
    }
    index += rect2raw_sample_col_num;

    // generate rect2rawSampleY from index to index + rect2raw_sample+row_num - 1
    for (int i = 0; i < rect2raw_sample_row_num; i++) {
        rect2rawSampleY.push_back(lut_info[index++] / 2);
    }
    index += rect2raw_sample_row_num;

    // if rect2rawSampleX > 2 * max_diff, then make it to rect2rawSampleX - 2^14
    for (int i = 0; i < rect2rawSampleX.size(); i++) {
        if (rect2rawSampleX[i] > 2 * max_diff) {
            rect2rawSampleX[i] = rect2rawSampleX[i] - pow(2, 14);
        }
    }

    // do the same for rect2rawSampleY but rect2rawSampleY - 2^13
    for (int i = 0; i < rect2rawSampleY.size(); i++) {
        if (rect2rawSampleY[i] > 2 * max_diff) {
            rect2rawSampleY[i] = rect2rawSampleY[i] - pow(2, 13);
        }
    }

    // if rect2rawSampleX < -max_diff, then make it to rect2rawSampleX + 2^14
    for (int i = 0; i < rect2rawSampleX.size(); i++) {
        if (rect2rawSampleX[i] < -max_diff) {
            rect2rawSampleX[i] = rect2rawSampleX[i] + pow(2, 13);
        }
    }

    // do the same for rect2rawSampleY but rect2rawSampleY + 2^13
    for (int i = 0; i < rect2rawSampleY.size(); i++) {
        if (rect2rawSampleY[i] < -max_diff) {
            rect2rawSampleY[i] = rect2rawSampleY[i] + pow(2, 13);
        }
    }

    // convert vector to cv::Mat for raw2rectSampleX, raw2rectSampleY, rect2rawSampleX, rect2rawSampleY
    cv::Mat raw2rectSampleX_mat(raw2rectSampleX.size(), 1, CV_64F, raw2rectSampleX.data());
    cv::Mat raw2rectSampleY_mat(raw2rectSampleY.size(), 1, CV_64F, raw2rectSampleY.data());
    cv::Mat rect2rawSampleX_mat(rect2rawSampleX.size(), 1, CV_64F, rect2rawSampleX.data());
    cv::Mat rect2rawSampleY_mat(rect2rawSampleY.size(), 1, CV_64F, rect2rawSampleY.data());

    // transpose raw2rectSampleX_mat and rect2rawSampleX_mat
    cv::transpose(raw2rectSampleX_mat, raw2rectSampleX_mat);
    cv::transpose(rect2rawSampleX_mat, rect2rawSampleX_mat);

    // repeat raw2rectSampleX_mat with dimension (rect2raw_sample_row_num, 1)
    cv::repeat(raw2rectSampleX_mat, rect2raw_sample_row_num, 1, raw2rectSampleX_mat);
    cv::repeat(raw2rectSampleY_mat, 1, raw2rect_sample_col_num, raw2rectSampleY_mat);

    // do the same for rect2rawSampleX_mat and rect2rawSampleY_mat
    cv::repeat(rect2rawSampleX_mat, 1, rect2raw_sample_row_num, rect2rawSampleX_mat);
    cv::repeat(rect2rawSampleY_mat, raw2rect_sample_col_num, 1, rect2rawSampleY_mat);

    int raw2rect_int_len = int_len;
    int raw2rect_frac_len = frac_len;
    int raw2rect_world_len = int_len + frac_len;
    // generate raw2rect_delta_sample from index to index + raw2rect_sample_row_num * raw2rect_sample_col_num - 1
    std::vector<double> raw2rect_delta_sample;
    for (int i = 0; i < raw2rect_sample_row_num * raw2rect_sample_col_num; i++) {
        raw2rect_delta_sample.push_back(lut_info[index++]);
    }

    // apply dec2bin to raw2rect_delta_sample with raw2rect_world_len * 2 bits
    // convert to binary with raw2rect_world_len * 2bits
    std::vector<std::string> raw2rect_delta_sample_bin;
    for (int i = 0; i < raw2rect_delta_sample.size(); i++) {
        raw2rect_delta_sample_bin.push_back(dec2bin(int(raw2rect_delta_sample[i]), raw2rect_world_len * 2));
    }
    index += raw2rect_sample_row_num * raw2rect_sample_col_num;

    std::vector<double> raw2rect_delta_sample_x, raw2rect_delta_sample_y;
    // convert binary to decimal for raw2rec_delta_sample_bin
    for (int i = 0; i < raw2rect_delta_sample_bin.size(); i++) {
        int _len = raw2rect_delta_sample_bin[i].length();
        // subsrting from 0 to raw2rect_world_len in raw2rect_delta_sample_bin
        std::string _suby = raw2rect_delta_sample_bin[i].substr(0, raw2rect_world_len);
        std::string _subx = raw2rect_delta_sample_bin[i].substr(raw2rect_world_len+1, _len);
        raw2rect_delta_sample_y.push_back(bin2dec(_suby) / pow(2, raw2rect_frac_len));
        raw2rect_delta_sample_x.push_back(bin2dec(_subx) / pow(2, raw2rect_frac_len));
    }

    // if raw2rect_delta_sample_y > 0.5*2^raw2rect_int_len-1, then make it to raw2rect_delta_sample_y - 2^raw2rect_int_len
    for (int i = 0; i < raw2rect_delta_sample_y.size(); i++) {
        if (raw2rect_delta_sample_y[i] > 0.5 * pow(2, raw2rect_int_len - 1)) {
            raw2rect_delta_sample_y[i] = raw2rect_delta_sample_y[i] - pow(2, raw2rect_int_len);
        }
    }
    // if raw2rect_delta_sample_x > 0.5*2^raw2rect_int_len-1, then make it to raw2rect_delta_sample_y - 2^raw2rect_int_len
    for (int i = 0; i < raw2rect_delta_sample_x.size(); i++) {
       
        if (raw2rect_delta_sample_x[i] > 0.5 * pow(2, raw2rect_int_len - 1)) {
            raw2rect_delta_sample_x[i] = raw2rect_delta_sample_x[i] - pow(2, raw2rect_int_len);
        }
    }
    // if raw2rect_delta_sample_y < -0.5*2^raw2rect_int_len, then make it to raw2rect_delta_sample_y + 2^raw2rect_int_len
    for (int i = 0; i < raw2rect_delta_sample_y.size(); i++) {
        if (raw2rect_delta_sample_y[i] < -0.5 * pow(2, raw2rect_int_len)) {
            raw2rect_delta_sample_y[i] = raw2rect_delta_sample_y[i] + pow(2, raw2rect_int_len);
        }
    }
    // if raw2rect_delta_sample_x < -0.5*2^raw2rect_int_len, then make it to raw2rect_delta_sample_x + 2^raw2rect_int_len
    for (int i = 0; i < raw2rect_delta_sample_x.size(); i++) {
        if (raw2rect_delta_sample_x[i] < -0.5 * pow(2, raw2rect_int_len)) {
            raw2rect_delta_sample_x[i] = raw2rect_delta_sample_x[i] + pow(2, raw2rect_int_len);
        }
    }

    // convert vector to cv::Mat for raw2rect_delta_sample_y and raw2rect_delta_sample_x
    cv::Mat raw2rect_delta_sample_x_mat(raw2rect_delta_sample_x.size(), 1, CV_64F, raw2rect_delta_sample_x.data());
    cv::Mat raw2rect_delta_sample_y_mat(raw2rect_delta_sample_y.size(), 1, CV_64F, raw2rect_delta_sample_y.data());

    cv::Mat raw2rect_delta_samplex = raw2rect_delta_sample_x_mat.reshape(0, raw2rect_sample_row_num);   // Mat::reshape(int cn, int rows=0) const
    cv::Mat raw2rect_delta_sampley = raw2rect_delta_sample_y_mat.reshape(0, raw2rect_sample_col_num);


    /**
     * do the same thing for rect to raw
    **/
    // get the length of int and frac
    int rect2raw_int_len = int_len;
    int rect2raw_frac_len = frac_len;
    int rect2raw_world_len = int_len + frac_len;
    std::vector<double> rect2raw_delta_sample;
    for(int i = 0; i < rect2raw_sample_row_num * rect2raw_sample_col_num - 1; i++) {
        rect2raw_delta_sample.push_back(lut_info[index++]);
    }
    index += rect2raw_sample_row_num * rect2raw_sample_col_num;

    std::vector<std::string> rect2raw_delta_sample_bin;
    for (int i = 0; i < rect2raw_delta_sample.size(); i++) {
        rect2raw_delta_sample_bin.push_back(dec2bin(int(rect2raw_delta_sample[i]), rect2raw_world_len * 2));
    }

    std::vector<double> rect2raw_delta_sample_x, rect2raw_delta_sample_y;
    // convert binary to decimal for rect2raw_delta_sample_bin
    for (int i = 0; i < rect2raw_delta_sample_bin.size(); i++) {
        int _len = rect2raw_delta_sample_bin[i].length();
        // subsrting from 0 to raw2rect_world_len in raw2rect_delta_sample_bin
        std::string _suby = rect2raw_delta_sample_bin[i].substr(0, rect2raw_world_len);
        std::string _subx = rect2raw_delta_sample_bin[i].substr(rect2raw_world_len+1, _len);
        rect2raw_delta_sample_y.push_back(bin2dec(_suby) / pow(2, rect2raw_frac_len));
        rect2raw_delta_sample_x.push_back(bin2dec(_subx) / pow(2, rect2raw_frac_len));
    }

    // if rect2raw_delta_sample_y > 0.5 * 2^rect2raw_int_len-1, then make it to rect2raw_delta_sample_y - 2^rect2raw_int_len
    for (int i = 0; i < rect2raw_delta_sample_y.size(); i++) {
        if (rect2raw_delta_sample_y[i] > 0.5 * pow(2, rect2raw_int_len - 1)) {
            rect2raw_delta_sample_y[i] = rect2raw_delta_sample_y[i] - pow(2, rect2raw_int_len);
        }
    }

    // same for rect_raw_delta_sample_x
    for (int i = 0; i < rect2raw_delta_sample_x.size(); i++) {
        if (rect2raw_delta_sample_x[i] > 0.5 * pow(2, rect2raw_int_len - 1)) {
            rect2raw_delta_sample_x[i] = rect2raw_delta_sample_x[i] - pow(2, rect2raw_int_len);
        }
    }

    // if rect2raw_delta_sample_y < -0.5*2^rect2raw_int_len, then make it to rect2raw_delta_sample_y + 2^rect2raw_int_len
    for (int i = 0; i < rect2raw_delta_sample_y.size(); i++) {
        if (rect2raw_delta_sample_y[i] < -0.5 * pow(2, rect2raw_int_len)) {
            rect2raw_delta_sample_y[i] = rect2raw_delta_sample_y[i] + pow(2, rect2raw_int_len);
        }
    }

    // same for rect_raw_delta_sample_x
    for (int i = 0; i < rect2raw_delta_sample_x.size(); i++) {
        if (rect2raw_delta_sample_x[i] < -0.5 * pow(2, rect2raw_int_len)) {
            rect2raw_delta_sample_x[i] = rect2raw_delta_sample_x[i] + pow(2, rect2raw_int_len);
        }
    }

    // convert vector to cv::Mat for rect2raw_dekta_sample_x and rect2raw_delta_sample_y
    cv::Mat rect2raw_delta_sample_x_mat(rect2raw_delta_sample_x.size(), 1, CV_64F, rect2raw_delta_sample_x.data());
    cv::Mat rect2raw_delta_sample_y_mat(rect2raw_delta_sample_y.size(), 1, CV_64F, rect2raw_delta_sample_y.data());

    cv::Mat rect2raw_delta_samplex = rect2raw_delta_sample_x_mat.reshape(0, rect2raw_sample_row_num);
    cv::Mat rect2raw_delta_sampley = rect2raw_delta_sample_y_mat.reshape(0, rect2raw_sample_col_num);

    // convert vector to cv::Mat for raw2rectSampleX, raw2rectSampleY, rect2rawSampleX, rect2rawSampleY
    cv::Mat raw2rectSamplex(raw2rect_sample_row_num, raw2rect_sample_col_num, CV_64F, raw2rectSampleX.data());
    cv::Mat raw2rectSampley(raw2rect_sample_row_num, raw2rect_sample_col_num, CV_64F, raw2rectSampleY.data());
    cv::Mat rect2rawSamplex(rect2raw_sample_row_num, rect2raw_sample_col_num, CV_64F, rect2rawSampleX.data());
    cv::Mat rect2rawSampley(rect2raw_sample_row_num, rect2raw_sample_col_num, CV_64F, rect2rawSampleY.data());


    cv::Mat Raw2RectMapX, Raw2RectMapY, Rect2RawMapX, Rect2RawMapY;
    cv::subtract(raw2rectSamplex, raw2rect_delta_samplex, Raw2RectMapX);
    cv::subtract(raw2rectSampley, raw2rect_delta_sampley, Raw2RectMapY);
    cv::subtract(rect2rawSamplex, rect2raw_delta_samplex, Rect2RawMapX);
    cv::subtract(rect2rawSampley, rect2raw_delta_sampley, Rect2RawMapY);

    int nr = row_num;
    int nc = col_num;
}