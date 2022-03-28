#include "utils.h"

Eigen::MatrixXd normc(Eigen::MatrixXd x) {
    Eigen::MatrixXd _norm(x.rows(), x.cols());
    if (x.sum() / x.size() >= 0) {
        for (int i = 0; i < x.cols(); i++) {
            _norm.col(i) = x.col(i) / x.col(i).maxCoeff();
        }
    }
    else {
        for (int j = 0; j < x.rows(); j++) {
            for (int k = 0; k < x.cols(); k++) {
                _norm(j, k) = (x(j, k) - x.col(j).minCoeff()) / (x.col(j).maxCoeff() - x.col(j).minCoeff());
            }
        }
    }
    return _norm;
}

void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y) {
    std::vector<int> t_x, t_y;
    for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
    for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);

    // need to transpose X
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}

cv::Mat comp_distortion_oulu(cv::Mat xd, cv::Mat k) {		//压缩失真,数学计算
	int kk;
	double k1, k2, k3, p1, p2;
	double row = xd.rows, col = xd.cols;
	cv::Mat  x(row, col, CV_64FC1);
	cv::Mat  x_square(row, col, CV_64FC1);	//x矩阵的2次幂
	cv::Mat  r_2(1, col, CV_64FC1);			//x矩阵的2次幂按列加
	cv::Mat  k_radial(1, col, CV_64FC1);
	cv::Mat  k_radial_vconcat(row, col, CV_64FC1);	//两个一维k_radial拼成二维
	cv::Mat  r_2_square(1, col, CV_64FC1);	//r_2矩阵的2次幂
	cv::Mat  r_2_cube(1, col, CV_64FC1);	//r_2矩阵的3次幂
	cv::Mat  delta_x(row, col, CV_64FC1);
	cv::Mat  x_0_square(1, col, CV_64FC1);	//x矩阵第一行的2次幂
	cv::Mat  x_1_square(1, col, CV_64FC1);	//x矩阵第一行的2次幂

	k1 = k.at<double>(0, 0);
	k2 = k.at<double>(1, 0);
	k3 = k.at<double>(4, 0);
	p1 = k.at<double>(2, 0);
	p2 = k.at<double>(3, 0);

	x = xd.clone();

	for (kk = 0; kk < 20; kk++)
	{
		cv::pow(x, 2, x_square);  //pow(src, x, dst)
		cv::reduce(x_square, r_2, 0, REDUCE_SUM); //reduce(src, dst, 0, type)0按列加，1按行加 219行结束
		cv::pow(r_2, 2, r_2_square);
		cv::pow(r_2, 3, r_2_cube);
		k_radial = 1 + k1 * r_2 + k2 * r_2_square + k3 * r_2_cube;

		cv::pow(x.rowRange(0, 1), 2, x_0_square);	//rowRange(start, end)按行取，mul对应位置元素相乘
		cv::pow(x.rowRange(1, 2), 2, x_1_square);
		cv::vconcat(2 * p1 * ((x.rowRange(0, 1)).mul(x.rowRange(1, 2))) + p2 * (r_2 + 2 * x_0_square),
			2 * p2 * ((x.rowRange(0, 1)).mul(x.rowRange(1, 2))) + p1 * (r_2 + 2 * x_1_square),
			delta_x);	//delta_x用cv::vconcat(src1, src2, dst)按列拼接矩阵
		cv::vconcat(k_radial, k_radial, k_radial_vconcat);
		x = (xd - delta_x) / k_radial_vconcat; //每个相同位置的元素相除,和Matlab点除功能相同
	}
	return x;
}

cv::Mat normalize_pixel(cv::Mat x_kk, cv::Mat fc, cv::Mat cc, cv::Mat kc, double alpha_c) {
    double row = x_kk.rows, col = x_kk.cols;
	cv::Mat xn(row, col, CV_64FC1), x_distort(row, col, CV_64FC1);	//2行864列double单通道矩阵

																	//用cv::vconcat(src1, src2, dst)按列拼接矩阵
	cv::vconcat((x_kk.rowRange(0, 1) - cc.at<double>(0, 0)) / fc.at<double>(0, 0),
		(x_kk.rowRange(1, 2) - cc.at<double>(1, 0)) / fc.at<double>(1, 0), x_distort);	//180行计算x_distort
	x_distort.rowRange(0, 1) = x_distort.rowRange(0, 1) - alpha_c * x_distort.rowRange(1, 2);
	xn = comp_distortion_oulu(x_distort, kc);
	return	xn;
}

cv::Mat Orig2Rect(cv::Mat pix, cv::Mat intrMatOld, cv::Mat intrMatNew, cv::Mat R, cv::Mat kc) { //原图到转换图的映射关系,和remapRect互为逆运算,使用的标定内
	cv::Mat pixUndist, pixUndistR, pixRect;
	cv::Mat pix_transpose, pixUndistHomo;

	cv::transpose(pix, pix_transpose);	//pix求转置 transpose(src, dst);
	cv::Mat input1 = (cv::Mat_<double>(2, 1) << intrMatOld.at<double>(0, 0), intrMatOld.at<double>(1, 1));
	cv::Mat input2 = (cv::Mat_<double>(2, 1) << intrMatOld.at<double>(0, 2), intrMatOld.at<double>(1, 2));
	pixUndist = normalize_pixel(pix_transpose, input1, input2, kc, 0);	//

	cv::Mat monesmat = cv::Mat::ones(1, pixUndist.cols, CV_64FC1);
	cv::vconcat(pixUndist, monesmat, pixUndistHomo);	//按列拼接矩阵
	pixUndistR = R * pixUndistHomo;
	pixRect = intrMatNew * pixUndistR;

	cv::vconcat(pixRect.rowRange(0, 1) / pixRect.rowRange(2, 3),
		pixRect.rowRange(1, 2) / pixRect.rowRange(2, 3), pixRect);  //用cv::vconcat(src1, src2, dst)按列拼接矩阵
	cv::transpose(pixRect, pixRect);	//pixRect转置
	return pixRect;
}