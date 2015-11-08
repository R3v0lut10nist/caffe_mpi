#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef MAX_INF
#define MAX_INF pow(2,63)-1
#endif

// caffe.proto > LayerParameter > WindowDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

template <typename Dtype>
WindowDataMultiContextLayer<Dtype>::~WindowDataMultiContextLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void WindowDataMultiContextLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2

  //window_file format: input_format = 1
  // repeated
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2 cls_eff_num [cls_eff_idx cls_eff_label]

  //window_file format: input_format = 2
  // repeated
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2 region_cls_eff_num [region_cls_eff_idx]

  LOG(INFO) << "Window data layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.window_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.window_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.window_data_param().fg_fraction() << std::endl
      << "  cache_images: "
      << this->layer_param_.window_data_param().cache_images() << std::endl
      << "  root_folder: "
      << this->layer_param_.window_data_param().root_folder();

  cache_images_ = this->layer_param_.window_data_param().cache_images();
  string root_folder = this->layer_param_.window_data_param().root_folder();


  const int window_input_format= this->layer_param_.window_data_param().input_format();
  const int window_cls_num=this->layer_param_.window_data_param().cls_num();
  const int window_region_cls_num=this->layer_param_.window_data_param().region_cls_num();
  const float window_rotation_degree=this->layer_param_.window_data_param().rotation_degree();

  const int window_image_start=this->layer_param_.window_data_param().image_start();
  const int window_image_end= (this->layer_param_.window_data_param().image_end() < 0 ) ?
   MAX_INF : this->layer_param_.window_data_param().image_end();
  
  CHECK_GE(window_image_end,window_image_start);

  LOG(INFO) << "Window data layer:" <<std::endl 
      <<"window_input_format" <<window_input_format <<" window_cls_num" <<window_cls_num << " window_region_cls_num" << window_region_cls_num;

  const bool prefetch_needs_rand =
      this->transform_param_.mirror() ||
      this->transform_param_.crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }

  std::ifstream infile(this->layer_param_.window_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.window_data_param().source() << std::endl;

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index, channels;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");

    image_index = image_index - window_image_start;
    
    if(image_index > window_image_end - window_image_start){
      break;
    }

    // read image path
    string image_path;
    infile >> image_path;
    image_path = root_folder + image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0];
    if(image_index >= 0)
    image_database_.push_back(std::make_pair(image_path, image_size));

    if (cache_images_  && image_index >= 0) {
      Datum datum;
      if (!ReadFileToDatum(image_path, &datum)) {
        LOG(ERROR) << "Could not open or find file " << image_path;
        return;
      }
      image_database_cache_.push_back(std::make_pair(image_path, datum));
    }
    // read each box
    int num_windows;
    infile >> num_windows;
    const float fg_threshold =
        this->layer_param_.window_data_param().fg_threshold();
    const float bg_threshold =
        this->layer_param_.window_data_param().bg_threshold();
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2;
      float overlap;
      vector<float> window_cls_eff(window_cls_num+1,0.); // 1 image_index
      window_cls_eff[0]=image_index;

      vector<float> window_region_cls_eff(window_region_cls_num+1,0); // 1 image_index
      window_region_cls_eff[0]=image_index;

      int cls_eff_num=0;
      int region_cls_eff_num=0;
      if(window_input_format==0){
        infile >> label >> overlap >> x1 >> y1 >> x2 >> y2;
      }else if(window_input_format==1){
        infile >> label >> overlap >> x1 >> y1 >> x2 >> y2 >>cls_eff_num;
        CHECK_GT(cls_eff_num,0);
        CHECK_GT(window_cls_num,0);
        int cls_eff_idx; float cls_eff_label;
        for(int cls_eff_num_idx=0;cls_eff_num_idx<cls_eff_num;++cls_eff_num_idx){
          infile >> cls_eff_idx >> cls_eff_label;
          CHECK_GT(window_cls_num,cls_eff_idx);
          window_cls_eff[cls_eff_idx+1]=cls_eff_label;
        }
      }else if(window_input_format==2){
        infile >> label >> overlap >> x1 >> y1 >> x2 >> y2 >>region_cls_eff_num;
        CHECK_GE(region_cls_eff_num,0)<< image_path;
        CHECK_GT(window_region_cls_num,0);
        int region_cls_eff_idx;
        for(int region_cls_eff_num_idx=0;region_cls_eff_num_idx<region_cls_eff_num;++region_cls_eff_num_idx){
          infile >> region_cls_eff_idx;
          CHECK_GE(window_region_cls_num,region_cls_eff_idx);
          window_region_cls_eff[region_cls_eff_idx+1]=1;
        }
      }else{
        LOG(FATAL) << "unknown window format";
      }

      vector<float> window(WindowDataMultiContextLayer::NUM);
      window[WindowDataMultiContextLayer::IMAGE_INDEX] = image_index;
      window[WindowDataMultiContextLayer::LABEL] = label;
      window[WindowDataMultiContextLayer::OVERLAP] = overlap;
      window[WindowDataMultiContextLayer::X1] = x1;
      window[WindowDataMultiContextLayer::Y1] = y1;
      window[WindowDataMultiContextLayer::X2] = x2;
      window[WindowDataMultiContextLayer::Y2] = y2;

      if(image_index < 0){
        continue;
      }
      // add window to foreground list or background list
      if (overlap >= fg_threshold) {
        int label = window[WindowDataMultiContextLayer::LABEL];
        CHECK_GT(label, 0);
        fg_windows_.push_back(window);
        label_hist.insert(std::make_pair(label, 0));
        label_hist[label]++;
        fg_windows_eff_labels_.push_back(window_cls_eff);
        fg_windows_region_eff_labels_.push_back(window_region_cls_eff);
      } else if (overlap < bg_threshold) {
        // background window, force label and overlap to 0
        window[WindowDataMultiContextLayer::LABEL] = 0;
        window[WindowDataMultiContextLayer::OVERLAP] = 0;
        bg_windows_.push_back(window);
        label_hist[0]++;
        bg_windows_eff_labels_.push_back(window_cls_eff);
        bg_windows_region_eff_labels_.push_back(window_region_cls_eff);
      }

      total_windows_.push_back(window);
      total_windows_eff_labels_.push_back(window_cls_eff);
      total_windows_region_eff_labels_.push_back(window_region_cls_eff);
    }

    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  } while (infile >> hashtag >> image_index);

  LOG(INFO) << "Number of images: " << image_index+1;

  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.window_data_param().multi_context_pad_size();

  LOG(INFO) << "Amount of rotation degree: "
      << this->layer_param_.window_data_param().multi_rotation_degree_size();

  multi_context_pad_.clear();
  for(int i=0; i< this->layer_param_.window_data_param().multi_context_pad_size();++i){
    multi_context_pad_.push_back(this->layer_param_.window_data_param().multi_context_pad(i));
    LOG(INFO) << "context padding: " << this->layer_param_.window_data_param().multi_context_pad(i);
  }
  CHECK_GT(multi_context_pad_.size(),0);

  multi_rotation_degree_.clear();
  if(this->layer_param_.window_data_param().multi_rotation_degree_size() > 0){
    for(int i=0; i< this->layer_param_.window_data_param().multi_rotation_degree_size();++i){
      multi_rotation_degree_.push_back(this->layer_param_.window_data_param().multi_rotation_degree(i));
      LOG(INFO) << "rotation degree: " << this->layer_param_.window_data_param().multi_rotation_degree(i);
    }
    CHECK_GT(multi_rotation_degree_.size(),0);
  }else{
    for(int i=0; i< multi_context_pad_.size();++i){
      multi_rotation_degree_.push_back(window_rotation_degree);
      LOG(INFO) << "rotation degree: " << window_rotation_degree;
    }
  }
  CHECK_EQ(multi_rotation_degree_.size(),multi_context_pad_.size()) << "context_pad size should equal as rotation degree size";


  LOG(INFO) << "Crop mode: "
      << this->layer_param_.window_data_param().crop_mode();

  // image
  const int crop_size = this->transform_param_.crop_size();
  CHECK_GT(crop_size, 0);
  const int batch_size = this->layer_param_.window_data_param().batch_size();
  top[0]->Reshape(batch_size * multi_context_pad_.size(), channels, crop_size, crop_size);
  this->prefetch_data_.Reshape(batch_size * multi_context_pad_.size(), channels, crop_size, crop_size);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();


  if(window_input_format==0){
    top[1]->Reshape(batch_size, 1, 1, 1);
    this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
  }else if(window_input_format==1){
    top[1]->Reshape(batch_size, window_cls_num, 1,1);
    this->prefetch_label_.Reshape(batch_size, window_cls_num,1,1);

    LOG(INFO) << "output data size: " << top[1]->num() << ","
        << top[1]->channels() << "," << top[1]->height() << ","
        << top[1]->width();
  }else if(window_input_format==2){
    top[1]->Reshape(batch_size, 1, 1, 1);
    this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

    top[2]->Reshape(batch_size, window_region_cls_num, 1, 1);
    this->prefetch_region_label_.Reshape(batch_size, window_region_cls_num, 1, 1);
  }
  // label
  //vector<int> label_shape(1, batch_size);
  //top[1]->Reshape(label_shape);
  //this->prefetch_label_.Reshape(label_shape);

  // data mean
  has_mean_file_ = this->transform_param_.has_mean_file();
  has_mean_values_ = this->transform_param_.mean_value_size() > 0;

  if (has_mean_values_) {
    if(has_mean_file_==true){
      LOG(INFO)<<"Cannot specify mean_file and mean_value at the same time";
      LOG(INFO)<<"Ignore mean_file and use mean_value";
    }
    for (int c = 0; c < this->transform_param_.mean_value_size(); ++c) {
      mean_values_.push_back(this->transform_param_.mean_value(c));
    }
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
     "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }else if (has_mean_file_) {
    const string& mean_file =
          this->transform_param_.mean_file();
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }

}

template <typename Dtype>
unsigned int WindowDataMultiContextLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// Thread fetching the data
template <typename Dtype>
void WindowDataMultiContextLayer<Dtype>::InternalThreadEntry() {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  Dtype* top_region_label = NULL;
  const Dtype scale = this->layer_param_.window_data_param().scale();
  const int batch_size = this->layer_param_.window_data_param().batch_size();
  
  const int crop_size = this->transform_param_.crop_size();
  const bool mirror = this->transform_param_.mirror();
  const float fg_fraction =
      this->layer_param_.window_data_param().fg_fraction();

  const bool random_order = this->layer_param_.window_data_param().random_order();

  Dtype* mean = NULL;
  int mean_off = 0;
  int mean_width = 0;
  int mean_height = 0;
  if (this->has_mean_file_) {
    mean = this->data_mean_.mutable_cpu_data();
    mean_off = (this->data_mean_.width() - crop_size) / 2;
    mean_width = this->data_mean_.width();
    mean_height = this->data_mean_.height();
  }
  cv::Size cv_crop_size(crop_size, crop_size);
  const string& crop_mode = this->layer_param_.window_data_param().crop_mode();

  const int window_input_format= this->layer_param_.window_data_param().input_format();
  const int window_cls_num=this->layer_param_.window_data_param().cls_num();
  const int window_region_cls_num=this->layer_param_.window_data_param().region_cls_num();


  const int multi_context_pad_size=this->layer_param_.window_data_param().multi_context_pad_size();
  CHECK_EQ(multi_context_pad_size, multi_context_pad_.size());

  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);

  if(window_input_format == 2){
     top_region_label = this->prefetch_region_label_.mutable_cpu_data();
    caffe_set(this->prefetch_region_label_.count(), Dtype(0), top_region_label);
  }

  const int num_fg = static_cast<int>(static_cast<float>(batch_size)
      * fg_fraction);
  const int num_samples[2] = { batch_size - num_fg, num_fg };

  int item_id = 0;
  cv::Mat cv_previous_img; int previous_img_index = -1;
  // sample from bg set then fg set
  for (int is_fg = 0; is_fg < 2; ++is_fg) {
    for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
      // sample a window
      timer.Start();
      unsigned int rand_index;
      vector<float> window;
      vector<float> window_cls_eff;
      vector<float> window_region_cls_eff;
      if(random_order == true){
        rand_index = PrefetchRand();
        window = (is_fg) ?
          fg_windows_[rand_index % fg_windows_.size()] :
          bg_windows_[rand_index % bg_windows_.size()];

        window_cls_eff = (is_fg) ?
          fg_windows_eff_labels_[rand_index % fg_windows_.size()] :
          bg_windows_eff_labels_[rand_index % bg_windows_.size()];

        window_region_cls_eff = (is_fg) ?
          fg_windows_region_eff_labels_[rand_index % fg_windows_.size()] :
          bg_windows_region_eff_labels_[rand_index % bg_windows_.size()];
      }
      else{
        rand_index = this->total_windows_idx_++ % this->total_windows_.size();
        CHECK_EQ(this->total_windows_.size(),this->total_windows_eff_labels_.size());
        CHECK_EQ(this->total_windows_.size(),this->total_windows_region_eff_labels_.size());

        window = this->total_windows_[rand_index];
        window_cls_eff = this->total_windows_eff_labels_[rand_index];
        window_region_cls_eff = this->total_windows_region_eff_labels_[rand_index];
      }

      CHECK_EQ(window[WindowDataMultiContextLayer<Dtype>::IMAGE_INDEX], window_cls_eff[0]);
      CHECK_EQ(window[WindowDataMultiContextLayer<Dtype>::IMAGE_INDEX], window_region_cls_eff[0]);

      bool do_mirror = mirror && PrefetchRand() % 2 && random_order;

      // load the image containing the window
      pair<std::string, vector<int> > image =
          image_database_[window[WindowDataMultiContextLayer<Dtype>::IMAGE_INDEX]];

      cv::Mat cv_img;
      if (this->cache_images_) {
        pair<std::string, Datum> image_cached =
          image_database_cache_[window[WindowDataMultiContextLayer<Dtype>::IMAGE_INDEX]];
        cv_img = DecodeDatumToCVMat(image_cached.second, true);
      } else {
        if(random_order == false && window[WindowDataMultiContextLayer<Dtype>::IMAGE_INDEX] == previous_img_index){
          cv_img = cv_previous_img.clone();
        }else{
          cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
        }
        if (!cv_img.data) {
          LOG(ERROR) << "Could not open or find file " << image.first;
          return;
        }
        if(random_order == false && window[WindowDataMultiContextLayer<Dtype>::IMAGE_INDEX] != previous_img_index){
          previous_img_index=window[WindowDataMultiContextLayer<Dtype>::IMAGE_INDEX];
          cv_previous_img = cv_img.clone();
        }
      }
      read_time += timer.MicroSeconds();
      timer.Start();
      const int channels = cv_img.channels();

      cv::Mat cv_img_clone=cv_img.clone();

      for(int multi_context_pad_idx=0; multi_context_pad_idx<multi_context_pad_size;++multi_context_pad_idx){

        cv_img=cv_img_clone.clone();
        float window_rotation_degree=multi_rotation_degree_[multi_context_pad_idx];
        int context_pad = multi_context_pad_[multi_context_pad_idx];

        // crop window out of image and warp it
        int x1 = window[WindowDataMultiContextLayer<Dtype>::X1];
        int y1 = window[WindowDataMultiContextLayer<Dtype>::Y1];
        int x2 = window[WindowDataMultiContextLayer<Dtype>::X2];
        int y2 = window[WindowDataMultiContextLayer<Dtype>::Y2];

        //LOG(INFO)<< "image: "<< image.first;
        //LOG(INFO)<< x1 << " "<< y1 <<" "<<" "<< x2 <<" "<<y2;

        int pad_w = 0;
        int pad_h = 0;
        if (context_pad > 0 || use_square) {
          // scale factor by which to expand the original region
          // such that after warping the expanded region to crop_size x crop_size
          // there's exactly context_pad amount of padding on each side
          Dtype context_scale = static_cast<Dtype>(crop_size) /
              static_cast<Dtype>(crop_size - 2*context_pad);

          // compute the expanded region
          Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
          Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
          Dtype center_x = static_cast<Dtype>(x1) + half_width;
          Dtype center_y = static_cast<Dtype>(y1) + half_height;
          if (use_square) {
            if (half_height > half_width) {
              half_width = half_height;
            } else {
              half_height = half_width;
            }
          }
          x1 = static_cast<int>(round(center_x - half_width*context_scale));
          x2 = static_cast<int>(round(center_x + half_width*context_scale));
          y1 = static_cast<int>(round(center_y - half_height*context_scale));
          y2 = static_cast<int>(round(center_y + half_height*context_scale));

          // the expanded region may go outside of the image
          // so we compute the clipped (expanded) region and keep track of
          // the extent beyond the image
          int unclipped_height = y2-y1+1;
          int unclipped_width = x2-x1+1;
          int pad_x1 = std::max(0, -x1);
          int pad_y1 = std::max(0, -y1);
          int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
          int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
          // clip bounds
          x1 = x1 + pad_x1;
          x2 = x2 - pad_x2;
          y1 = y1 + pad_y1;
          y2 = y2 - pad_y2;
          CHECK_GT(x1, -1);
          CHECK_GT(y1, -1);
          CHECK_LT(x2, cv_img.cols);
          CHECK_LT(y2, cv_img.rows);

          int clipped_height = y2-y1+1;
          int clipped_width = x2-x1+1;

          // scale factors that would be used to warp the unclipped
          // expanded region
          Dtype scale_x =
              static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
          Dtype scale_y =
              static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

          // size to warp the clipped expanded region to
          cv_crop_size.width =
              static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
          cv_crop_size.height =
              static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
          pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
          pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
          pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
          pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

          pad_h = pad_y1;
          // if we're mirroring, we mirror the padding too (to be pedantic)
          if (do_mirror) {
            pad_w = pad_x2;
          } else {
            pad_w = pad_x1;
          }

          // ensure that the warped, clipped region plus the padding fits in the
          // crop_size x crop_size image (it might not due to rounding)
          if (pad_h + cv_crop_size.height > crop_size) {
            cv_crop_size.height = crop_size - pad_h;
          }
          if (pad_w + cv_crop_size.width > crop_size) {
            cv_crop_size.width = crop_size - pad_w;
          }
        }

        cv::Mat cv_cropped_img;

        if(window_rotation_degree !=0){
        int cv_rot_img_x1=std::min(std::min(-cv_img.cols/2*cos(window_rotation_degree*PI/180)-cv_img.rows/2*sin(window_rotation_degree*PI/180)
          ,-cv_img.cols/2*cos(window_rotation_degree*PI/180)+cv_img.rows/2*sin(window_rotation_degree*PI/180))
        ,std::min(cv_img.cols/2*cos(window_rotation_degree*PI/180)-cv_img.rows/2*sin(window_rotation_degree*PI/180)
          ,cv_img.cols/2*cos(window_rotation_degree*PI/180)+cv_img.rows/2*sin(window_rotation_degree*PI/180)));
        
        int cv_rot_img_x2=std::max(std::max(-cv_img.cols/2*cos(window_rotation_degree*PI/180)-cv_img.rows/2*sin(window_rotation_degree*PI/180)
          ,-cv_img.cols/2*cos(window_rotation_degree*PI/180)+cv_img.rows/2*sin(window_rotation_degree*PI/180))
        ,std::max(cv_img.cols/2*cos(window_rotation_degree*PI/180)-cv_img.rows/2*sin(window_rotation_degree*PI/180)
          ,cv_img.cols/2*cos(window_rotation_degree*PI/180)+cv_img.rows/2*sin(window_rotation_degree*PI/180)));
        
        int cv_rot_img_y1=std::min(std::min(-cv_img.cols/2*sin(window_rotation_degree*PI/180)+cv_img.rows/2*cos(window_rotation_degree*PI/180)
          ,-cv_img.cols/2*sin(window_rotation_degree*PI/180)-cv_img.rows/2*cos(window_rotation_degree*PI/180))
        ,std::min(cv_img.cols/2*sin(window_rotation_degree*PI/180)+cv_img.rows/2*cos(window_rotation_degree*PI/180)
          ,cv_img.cols/2*sin(window_rotation_degree*PI/180)-cv_img.rows/2*cos(window_rotation_degree*PI/180)));
        
        int cv_rot_img_y2=std::max(std::max(-cv_img.cols/2*sin(window_rotation_degree*PI/180)+cv_img.rows/2*cos(window_rotation_degree*PI/180)
          ,-cv_img.cols/2*sin(window_rotation_degree*PI/180)-cv_img.rows/2*cos(window_rotation_degree*PI/180))
        ,std::max(cv_img.cols/2*sin(window_rotation_degree*PI/180)+cv_img.rows/2*cos(window_rotation_degree*PI/180)
          ,cv_img.cols/2*sin(window_rotation_degree*PI/180)-cv_img.rows/2*cos(window_rotation_degree*PI/180)));
        
        cv::Point window_rotation_center = cv::Point(cv_img.cols/2,cv_img.rows/2);
        float window_rotation_scale= 1.0; 
        cv::Mat rot_mat(2,3,CV_32FC1);
        rot_mat=cv::getRotationMatrix2D(window_rotation_center,window_rotation_degree,window_rotation_scale);
        int cv_rot_img_padx=std::max(0,cv_rot_img_x2-cv_img.cols/2);
        int cv_rot_img_pady=std::max(0,cv_rot_img_y2-cv_img.rows/2);
        rot_mat.at<double>(0,2)+=cv_rot_img_padx;
        rot_mat.at<double>(1,2)+=cv_rot_img_pady;

        cv::Mat cv_rot_img(cv_rot_img_y2-cv_rot_img_y1,cv_rot_img_x2-cv_rot_img_x1,cv_img.type());
        cv::warpAffine(cv_img, cv_rot_img, rot_mat, cv_rot_img.size());

        /*char save_filename[256];
        std::sprintf(save_filename,"test_%d.jpg",window[WindowDataMultiContextLayer<Dtype>::IMAGE_INDEX]);
        cv::imwrite(save_filename,cv_img);
        std::sprintf(save_filename,"test_rotate_%d.jpg",window[WindowDataMultiContextLayer<Dtype>::IMAGE_INDEX]);
        cv::imwrite(save_filename,cv_rot_img);*/
        

        int window_rotate_x1=std::min(std::min((x1-cv_img.cols/2)*cos(window_rotation_degree*PI/180)-(cv_img.rows/2-y1)*sin(window_rotation_degree*PI/180)
          ,(x1-cv_img.cols/2)*cos(window_rotation_degree*PI/180)-(cv_img.rows/2-y2)*sin(window_rotation_degree*PI/180))
        ,std::min((x2-cv_img.cols/2)*cos(window_rotation_degree*PI/180)-(cv_img.rows/2-y1)*sin(window_rotation_degree*PI/180)
          ,(x2-cv_img.cols/2)*cos(window_rotation_degree*PI/180)-(cv_img.rows/2-y2)*sin(window_rotation_degree*PI/180)));
        window_rotate_x1+=(cv_rot_img_x2-cv_rot_img_x1)/2;
        window_rotate_x1=std::max(0,window_rotate_x1);
        
        int window_rotate_x2=std::max(std::max((x1-cv_img.cols/2)*cos(window_rotation_degree*PI/180)-(cv_img.rows/2-y1)*sin(window_rotation_degree*PI/180)
          ,(x1-cv_img.cols/2)*cos(window_rotation_degree*PI/180)-(cv_img.rows/2-y2)*sin(window_rotation_degree*PI/180))
        ,std::max((x2-cv_img.cols/2)*cos(window_rotation_degree*PI/180)-(cv_img.rows/2-y1)*sin(window_rotation_degree*PI/180)
          ,(x2-cv_img.cols/2)*cos(window_rotation_degree*PI/180)-(cv_img.rows/2-y2)*sin(window_rotation_degree*PI/180)));
        window_rotate_x2+=(cv_rot_img_x2-cv_rot_img_x1)/2;
        window_rotate_x2=std::min((cv_rot_img_x2-cv_rot_img_x1)-1,window_rotate_x2);
        
        int window_rotate_y2=std::min(std::min((x1-cv_img.cols/2)*sin(window_rotation_degree*PI/180)+(cv_img.rows/2-y1)*cos(window_rotation_degree*PI/180)
          ,(x1-cv_img.cols/2)*sin(window_rotation_degree*PI/180)+(cv_img.rows/2-y2)*cos(window_rotation_degree*PI/180))
        ,std::min((x2-cv_img.cols/2)*sin(window_rotation_degree*PI/180)+(cv_img.rows/2-y1)*cos(window_rotation_degree*PI/180)
          ,(x2-cv_img.cols/2)*sin(window_rotation_degree*PI/180)+(cv_img.rows/2-y2)*cos(window_rotation_degree*PI/180)));
        window_rotate_y2=(cv_rot_img_y2-cv_rot_img_y1)/2-window_rotate_y2;
        window_rotate_y2=std::min((cv_rot_img_y2-cv_rot_img_y1)-1,window_rotate_y2);
        
        int window_rotate_y1=std::max(std::max((x1-cv_img.cols/2)*sin(window_rotation_degree*PI/180)+(cv_img.rows/2-y1)*cos(window_rotation_degree*PI/180)
          ,(x1-cv_img.cols/2)*sin(window_rotation_degree*PI/180)+(cv_img.rows/2-y2)*cos(window_rotation_degree*PI/180))
        ,std::max((x2-cv_img.cols/2)*sin(window_rotation_degree*PI/180)+(cv_img.rows/2-y1)*cos(window_rotation_degree*PI/180)
          ,(x2-cv_img.cols/2)*sin(window_rotation_degree*PI/180)+(cv_img.rows/2-y2)*cos(window_rotation_degree*PI/180)));
        window_rotate_y1=(cv_rot_img_y2-cv_rot_img_y1)/2-window_rotate_y1;
        window_rotate_y1=std::max(0,window_rotate_y1);

        //LOG(INFO)<<window_rotate_x1<<" "<<window_rotate_x2<<" "<<window_rotate_y1<<" "<<window_rotate_y2;
        //LOG(INFO)<<(cv_rot_img_x2-cv_rot_img_x1)<<" "<<(cv_rot_img_y2-cv_rot_img_y1);

        //cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
        //cv::Mat cv_cropped_img = cv_img(roi);
        //cv::resize(cv_cropped_img, cv_cropped_img,
        //    cv_crop_size, 0, 0, cv::INTER_LINEAR);
        //std::sprintf(save_filename,"test_crop_%d.jpg",window[WindowDataMultiContextLayer<Dtype>::IMAGE_INDEX]);
        //cv::imwrite(save_filename,cv_cropped_img);
        
        cv::Rect window_rotate_roi(window_rotate_x1, window_rotate_y1, window_rotate_x2-window_rotate_x1+1, window_rotate_y2-window_rotate_y1+1);
        cv_cropped_img = cv_rot_img(window_rotate_roi);
        cv::resize(cv_cropped_img, cv_cropped_img,
            cv_crop_size, 0, 0, cv::INTER_LINEAR);
        //std::sprintf(save_filename,"test_crop_rotate_%d.jpg",window[WindowDataMultiContextLayer<Dtype>::IMAGE_INDEX]);
        //cv::imwrite(save_filename,cv_cropped_rot_img);
        }else{
          cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
          cv_cropped_img = cv_img(roi);
          cv::resize(cv_cropped_img, cv_cropped_img,
              cv_crop_size, 0, 0, cv::INTER_LINEAR);
        }

        /*char save_filename[256];
        std::sprintf(save_filename,"test_crop_%f_coord_%f_%f_%f_%f_pad_%d_rotate_%f.jpg",window[WindowDataMultiContextLayer<Dtype>::IMAGE_INDEX],
          window[WindowDataMultiContextLayer<Dtype>::X1],window[WindowDataMultiContextLayer<Dtype>::Y1],
          window[WindowDataMultiContextLayer<Dtype>::X2],window[WindowDataMultiContextLayer<Dtype>::Y2],context_pad,window_rotation_degree);
        cv::imwrite(save_filename,cv_cropped_img);*/

      // horizontal flip at random
        if (do_mirror) {
          cv::flip(cv_cropped_img, cv_cropped_img, 1);
        }

        // copy the warped window into top_data
        for (int h = 0; h < cv_cropped_img.rows; ++h) {
          const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
          int img_index = 0;
          for (int w = 0; w < cv_cropped_img.cols; ++w) {
            for (int c = 0; c < channels; ++c) {
              int top_index = (((item_id+multi_context_pad_idx*batch_size) * channels + c) * crop_size + h + pad_h)
                       * crop_size + w + pad_w;
              // int top_index = (c * height + h) * width + w;
              Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
              if (this->has_mean_file_) {
                int mean_index = (c * mean_height + h + mean_off + pad_h)
                             * mean_width + w + mean_off + pad_w;
                top_data[top_index] = (pixel - mean[mean_index]) * scale;
              } else {
                if (this->has_mean_values_) {
                  top_data[top_index] = (pixel - this->mean_values_[c]) * scale;
                } else {
                  top_data[top_index] = pixel * scale;
                }
              }
            }
          }
        }
        trans_time += timer.MicroSeconds();
        // get window label
        //top_label[item_id] = window[WindowDataMultiContextLayer<Dtype>::LABEL];
        if(window_input_format==0){
          top_label[item_id] = window[WindowDataMultiContextLayer<Dtype>::LABEL];
        }else if(window_input_format==1){
          for(int cls_eff_idx=0;cls_eff_idx<window_cls_num;++cls_eff_idx){
            top_label[item_id*window_cls_num+cls_eff_idx]=window_cls_eff[cls_eff_idx+1];
          }
        }else if(window_input_format==2){
          top_label[item_id] = window[WindowDataMultiContextLayer<Dtype>::LABEL];

          for(int region_cls_eff_idx=0;region_cls_eff_idx<window_region_cls_num;++region_cls_eff_idx){
            top_region_label[item_id*window_region_cls_num+region_cls_eff_idx]=window_region_cls_eff[region_cls_eff_idx+1];
          }
        }

      }

      item_id++;
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(WindowDataMultiContextLayer);
REGISTER_LAYER_CLASS(WindowDataMultiContext);

}  // namespace caffe
