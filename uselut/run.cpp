#include "src/lut_parser.h"
#include "src/use_lut.h"

#include <memory>

int main() {
  const int row = 1080;
  const int col = 1920;
  const std::string lut_file =
      "D:/rectify_tools/cpp/data/output/LutDecRGB_1920_1080.txt";
  const std::string in_path = "D:/rectify_tools/cpp/data/output/input1920.yuv";
  const std::string out_path = "D:/rectify_tools/cpp/data/output/rectY1920.png";
  const int int_len = 9;
  const int frac_len = 5;

  std::shared_ptr<useLutInitParams> params =
      initialize(col, row, int_len, frac_len, lut_file, in_path, out_path);

  use_lut(params, true, false);
  return 0;
}