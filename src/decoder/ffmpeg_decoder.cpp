#include "lmp/decoder/ffmpeg_decoder.hpp"

#include <stdexcept>
#include <utility>

#if LMP_HAS_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#endif

namespace lmp::decoder {

#if LMP_HAS_FFMPEG
namespace {

struct FormatDeleter {
  void operator()(AVFormatContext *context) const noexcept {
    if (context != nullptr) {
      avformat_close_input(&context);
    }
  }
};

struct CodecDeleter {
  void operator()(AVCodecContext *context) const noexcept {
    avcodec_free_context(&context);
  }
};

struct FrameDeleter {
  void operator()(AVFrame *frame) const noexcept { av_frame_free(&frame); }
};

struct PacketDeleter {
  void operator()(AVPacket *packet) const noexcept { av_packet_free(&packet); }
};

struct SwsDeleter {
  void operator()(SwsContext *context) const noexcept {
    sws_freeContext(context);
  }
};

std::string ffmpeg_error(int code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE]{};
  av_strerror(code, buffer, sizeof(buffer));
  return buffer;
}

} // namespace

class FfmpegDecoder::Impl {
public:
  Impl(std::string input_url, std::uint32_t width, std::uint32_t height)
      : input_url_(std::move(input_url)), width_(width), height_(height) {
    AVDictionary *options = nullptr;
    av_dict_set(&options, "fflags", "nobuffer", 0);
    av_dict_set(&options, "flags", "low_delay", 0);
    av_dict_set(&options, "probesize", "32", 0);
    av_dict_set(&options, "analyzeduration", "0", 0);

    AVFormatContext *raw_format = nullptr;
    const auto open_result =
        avformat_open_input(&raw_format, input_url_.c_str(), nullptr, &options);
    av_dict_free(&options);
    if (open_result < 0) {
      throw std::runtime_error("cannot open FFmpeg input: " +
                               ffmpeg_error(open_result));
    }
    format_.reset(raw_format);

    const auto stream_result =
        avformat_find_stream_info(format_.get(), nullptr);
    if (stream_result < 0) {
      throw std::runtime_error("cannot read FFmpeg stream info: " +
                               ffmpeg_error(stream_result));
    }

    const auto best_stream = av_find_best_stream(
        format_.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (best_stream < 0) {
      throw std::runtime_error("FFmpeg input does not contain a video stream");
    }
    stream_index_ = best_stream;

    const auto *parameters = format_->streams[stream_index_]->codecpar;
    const auto *codec = avcodec_find_decoder(parameters->codec_id);
    if (codec == nullptr) {
      throw std::runtime_error("FFmpeg video decoder not found");
    }

    codec_.reset(avcodec_alloc_context3(codec));
    if (!codec_) {
      throw std::runtime_error("cannot allocate FFmpeg codec context");
    }
    const auto copy_result =
        avcodec_parameters_to_context(codec_.get(), parameters);
    if (copy_result < 0) {
      throw std::runtime_error("cannot copy FFmpeg codec parameters: " +
                               ffmpeg_error(copy_result));
    }
    codec_->flags |= AV_CODEC_FLAG_LOW_DELAY;
    const auto codec_result = avcodec_open2(codec_.get(), codec, nullptr);
    if (codec_result < 0) {
      throw std::runtime_error("cannot open FFmpeg codec: " +
                               ffmpeg_error(codec_result));
    }

    decoded_.reset(av_frame_alloc());
    converted_.reset(av_frame_alloc());
    packet_.reset(av_packet_alloc());
    if (!decoded_ || !converted_ || !packet_) {
      throw std::runtime_error("cannot allocate FFmpeg frame/packet");
    }

    converted_bytes_.resize(static_cast<std::size_t>(width_) * height_ * 3U);
    av_image_fill_arrays(converted_->data, converted_->linesize,
                         converted_bytes_.data(), AV_PIX_FMT_RGB24,
                         static_cast<int>(width_), static_cast<int>(height_),
                         1);
  }

  frame::Frame read_frame() {
    while (true) {
      const auto receive_result =
          avcodec_receive_frame(codec_.get(), decoded_.get());
      if (receive_result == 0) {
        convert_current_frame();
        return frame::Frame{
            width_,
            height_,
            frame::PixelFormat::Rgb,
            converted_bytes_,
            std::vector<std::size_t>{static_cast<std::size_t>(width_) * 3U},
            frame::Frame::Clock::now()};
      }
      if (receive_result != AVERROR(EAGAIN)) {
        throw std::runtime_error("cannot decode FFmpeg frame: " +
                                 ffmpeg_error(receive_result));
      }

      const auto read_result = av_read_frame(format_.get(), packet_.get());
      if (read_result < 0) {
        throw std::runtime_error("cannot read FFmpeg packet: " +
                                 ffmpeg_error(read_result));
      }
      if (packet_->stream_index == stream_index_) {
        const auto send_result =
            avcodec_send_packet(codec_.get(), packet_.get());
        av_packet_unref(packet_.get());
        if (send_result < 0) {
          throw std::runtime_error("cannot send FFmpeg packet: " +
                                   ffmpeg_error(send_result));
        }
      } else {
        av_packet_unref(packet_.get());
      }
    }
  }

  bool is_open() const noexcept {
    return format_ != nullptr && codec_ != nullptr;
  }

private:
  void convert_current_frame() {
    sws_.reset(sws_getCachedContext(
        sws_.release(), decoded_->width, decoded_->height,
        static_cast<AVPixelFormat>(decoded_->format), static_cast<int>(width_),
        static_cast<int>(height_), AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR, nullptr,
        nullptr, nullptr));
    if (!sws_) {
      throw std::runtime_error("cannot create FFmpeg scaler");
    }
    sws_scale(sws_.get(), decoded_->data, decoded_->linesize, 0,
              decoded_->height, converted_->data, converted_->linesize);
  }

  std::string input_url_;
  std::uint32_t width_;
  std::uint32_t height_;
  int stream_index_{-1};
  std::unique_ptr<AVFormatContext, FormatDeleter> format_;
  std::unique_ptr<AVCodecContext, CodecDeleter> codec_;
  std::unique_ptr<AVFrame, FrameDeleter> decoded_;
  std::unique_ptr<AVFrame, FrameDeleter> converted_;
  std::unique_ptr<AVPacket, PacketDeleter> packet_;
  std::unique_ptr<SwsContext, SwsDeleter> sws_;
  std::vector<std::uint8_t> converted_bytes_;
};
#else
class FfmpegDecoder::Impl {
public:
  Impl(std::string input_url, std::uint32_t width, std::uint32_t height) {
    static_cast<void>(input_url);
    static_cast<void>(width);
    static_cast<void>(height);
    throw std::runtime_error("FFmpeg support is not compiled in. Install "
                             "FFmpeg development packages "
                             "and rebuild.");
  }
  frame::Frame read_frame() { throw std::runtime_error("FFmpeg unavailable"); }
  bool is_open() const noexcept { return false; }
};
#endif

FfmpegDecoder::FfmpegDecoder(std::string input_url, std::uint32_t width,
                             std::uint32_t height)
    : impl_(std::make_unique<Impl>(std::move(input_url), width, height)) {}

FfmpegDecoder::~FfmpegDecoder() = default;

FfmpegDecoder::FfmpegDecoder(FfmpegDecoder &&) noexcept = default;

FfmpegDecoder &FfmpegDecoder::operator=(FfmpegDecoder &&) noexcept = default;

frame::Frame FfmpegDecoder::read_frame() { return impl_->read_frame(); }

bool FfmpegDecoder::is_open() const noexcept { return impl_->is_open(); }

} // namespace lmp::decoder
