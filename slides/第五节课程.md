# 自制深度学习推理框架-第六课-Max Pooling算子的实现

## 课程空间

视频课程：https://space.bilibili.com/1822828582

项目主页：[https://github.com/zjhellofss/KuiperInfer](https://github.com/zjhellofss/KuiperInfer)  欢迎点赞(star)和pr, 万分感谢大家.

## Max Pooling算子的定义

池化层在深度学习网络中的作用一般是用来缓解卷积层对位置的过度敏感性。池化层每次对输入数据的一个固定形状窗口(池化窗口的大小为`pooling height`, `pooling width`)中的元素计算输出，池化层直接计算池化窗口内元素的最大值或者平均值，因此该运算也分别叫做最大池化或平均池化。

在我们本节课要讲的二维最大池化中，池化窗口从输入数组的最左上方开始，按从左往右、从上往下的顺序，依次在输入数组上滑动(滑动的幅度被称为stride)。当池化窗口滑动到某一位置时，窗口中的输入**子数组的最大值**即输出数组中相应位置的元素。  

![](https://pica.zhimg.com/80/v2-f95358af1c14ce72ff426598753d2eeb_1440w.png?source=d16d100b)

图1展示了池化窗口形状为 `2×2` 的最大池化，阴影部分为第一个输出元素及其计算所使用的输入元素。

输出数组的高和宽分别为2，其中的4个元素由取最大值运算 max 得出。 如下公式所示，池化操作的步骤依次为从左到右，从上到下，每次向下移动的步长为`stride height`, 向右移动的步长为`stride width`. 进行池化操作元素的数量由`pooling height`和`pooling width`所组成的`2×2`的窗口所决定。
$$
max(0,1,3,4)=4\\ max(1,2,4,5)=5\\ max(3,4,6,7)=7\\ max(4,5,7,8)=8\\
$$

##  Max Pooling Operator的实现

```cpp
class MaxPoolingOp : public Operator {
 public:
  explicit MaxPoolingOp(uint32_t pooling_h, uint32_t pooling_w, uint32_t stride_h,
                        uint32_t stride_w, uint32_t padding_h, uint32_t padding_w);

  void set_pooling_h(uint32_t pooling_height);
  void set_pooling_w(uint32_t pooling_width);

  void set_stride_w(uint32_t stride_width);
  void set_stride_h(uint32_t stride_height);

  void set_padding_h(uint32_t padding_height);
  void set_padding_w(uint32_t padding_width);

  uint32_t padding_height() const;
  uint32_t padding_width() const;

  uint32_t stride_width() const;
  uint32_t stride_height() const;

  uint32_t pooling_height() const;
  uint32_t pooling_width() const;
 private:
  uint32_t pooling_h_; // 池化核高度大小
  uint32_t pooling_w_; // 池化核宽度大小
  uint32_t stride_h_;  // 高度上的步长
  uint32_t stride_w_;  // 宽度上的步长
  uint32_t padding_h_; // 高度上的填充
  uint32_t padding_w_; // 宽度上的填充
};
```

可以看到如上的Operator中,有6个类内属性，分别对应着我们第一节中讲过的步长(stride), 池化核(pooling)以及在池化前对边缘的扩充，以下我们在分别讲讲：

1. `stride`: 池化核每次移动的步长
2. `pooling`: 池化核的大小
3. `padding`: 对输入特征图的边缘扩充

如下图2是pad(padding值为1)后输入特征图的池化操作(池化核为2):

![](https://pic1.zhimg.com/80/v2-c8e69f6ee03b8bc266a88f18b0cc4f0e_1440w.png?source=d16d100b)
$$
 max(-\infty,-\infty,-\infty,0) =0\\ max(-\infty,-\infty,0,1) =1\\ max(-\infty,-\infty,1,2) =2\\ max(-\infty,-\infty,2,-\infty) =2\\ max(-\infty,0,-\infty,3) =3\\ max(0,1,3,4) =4\\   
$$

## Max Pooling Layer的实现

```cpp
MaxPoolingLayer::MaxPoolingLayer(const std::shared_ptr<Operator> &op) : Layer("maxpooling") {
  CHECK(op->op_type_ == OpType::kOperatorMaxPooling) << "Operator has a wrong type: " << int(op->op_type_);
  MaxPoolingOp *max_pooling_op = dynamic_cast<MaxPoolingOp *>(op.get());

  CHECK(max_pooling_op != nullptr) << "MaxPooling operator is empty";
  this->op_ = std::make_unique<MaxPoolingOp>(*max_pooling_op);
}

void MaxPoolingLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                               std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->op_type_ == OpType::kOperatorMaxPooling);
  CHECK(!inputs.empty());
  const uint32_t padding_h = this->op_->padding_height();
  const uint32_t padding_w = this->op_->padding_width();
  const uint32_t kernel_h = this->op_->pooling_height();
  const uint32_t kernel_w = this->op_->pooling_width();
  const uint32_t stride_h = this->op_->stride_height();
  const uint32_t stride_w = this->op_->stride_width();

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input_data_ = inputs.at(i)->Clone();
    input_data_->Padding({padding_h, padding_h, padding_w, padding_w}, std::numeric_limits<float>::lowest());
    const uint32_t input_h = input_data_->rows();
    const uint32_t input_w = input_data_->cols();
    const uint32_t input_c = input_data_->channels();
    const uint32_t output_c = input_c;

    const uint32_t output_h = uint32_t(std::floor((input_h - kernel_h) / stride_h + 1));
    const uint32_t output_w = uint32_t(std::floor((input_w - kernel_w) / stride_w + 1));
    CHECK(output_w > 0 && output_h > 0);

    std::shared_ptr<Tensor<float>> output_data = std::make_shared<Tensor<float>>(output_c, output_h, output_w);
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat &input_channel = input_data_->at(ic);
      arma::fmat &output_channel = output_data->at(ic);
      for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
        for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w) {
          const arma::fmat &region = input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
          output_channel.at(int(r / stride_h), int(c / stride_w)) = region.max();
        }
      }
    }
    outputs.push_back(output_data);
  }
}

std::shared_ptr<Layer> MaxPoolingLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
  CHECK(op->op_type_ == OpType::kOperatorMaxPooling);
  std::shared_ptr<Layer> max_layer = std::make_sh了ared<MaxPoolingLayer>(op);
  return max_layer;
}

LayerRegistererWrapper kMaxPoolingLayer(OpType::kOperatorMaxPooling, MaxPoolingLayer::CreateInstance);
```

```cpp
void MaxPoolingLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                               std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->op_type_ == OpType::kOperatorMaxPooling);
  CHECK(!inputs.empty());
}
```

我们重点来看`Forwards`函数, 首先判断输入是否为空并获得池化操作相关的属性值(原本存放在op中).

计算池化后的输出特征图大小, 公式为：
$$
output\,height = \lfloor \frac{input\,height -kernel\,height }{stride\, height}+1 \rfloor\\
output\,width = \lfloor \frac{input\,width -kernel\,width }{stride\, width}+1 \rfloor
$$

```cpp
 for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input_data_ = inputs.at(i)->Clone();
    input_data_->Padding({padding_h, padding_h, padding_w, padding_w}, std::numeric_limits<float>::lowest());
```

如上的过程表示对输入的特征图四周进行填充，填充的大小由于`padding_w`和`padding_h`决定。这两个Layer计算时候的属性由op中得到，也就是说`padding_w`和`padding_h`存放在`this->op`中，` this->op_ = std::make_unique<MaxPoolingOp>(*max_pooling_op);`

```cpp
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input_data_ = inputs.at(i)->Clone();
    input_data_->Padding({padding_h, padding_h, padding_w, padding_w}, std::numeric_limits<float>::lowest());
    const uint32_t input_h = input_data_->rows();
    const uint32_t input_w = input_data_->cols();
    const uint32_t input_c = input_data_->channels();
    const uint32_t output_c = input_c;

    const uint32_t output_h = uint32_t(std::floor((input_h - kernel_h) / stride_h + 1));
    const uint32_t output_w = uint32_t(std::floor((input_w - kernel_w) / stride_w + 1));
    CHECK(output_w > 0 && output_h > 0);
```

如上的过程表示根据输入的特征图大小`input_h`和`input_w`来计算对应的输出特征值大小`output_h`和`output_w`. 计算的公式如上文所示。如果输入的特征数据`input_data_`有填充，则根据填充数据的输入大小来计算对应的输出大小。

```cpp
for (uint32_t i = 0; i < batch_size; ++i) {
  ...
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat &input_channel = input_data_->at(ic);
      arma::fmat &output_channel = output_data->at(ic);
      for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
        for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w) {
          const arma::fmat &region = input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
          output_channel.at(int(r / stride_h), int(c / stride_w)) = region.max();
        }
      }
    }
}
```

`for(uint32_t ic =0; ic < input_c;++ic) `表示对输入的特征图进行逐通道的池化操作, 设当前进行操作的输入特征图通道为`input_channel`, 池化后的输出特征图放置于`output_channel`中。池化的过程如下公式所描述：
$$
output \,value = max([r:r+kernel \,height -1,c:c+kernel\,width -1])\\
$$
在上述的代码中`region`表示当前输入特征数据需要进行池化的部分，对应于公式中`[r:r+kernel height -1,c:c+kernel width -1]`

中的数据。输入特征的数据是逐个通道进行处理（池化操作）的，从`ic = 0`到`ic = input_channel - 1`, 当前池化的数据保存在`region`中。

`input_channel.submat(r, c, r + kernel_h -1, c + kernel_w -1)`取得一个池化区域内的所有元素，随后使用`region.max()`取得区域内(`kernel_h`和`kernel_w`组成的范围)的最大值, 并且每次区域移动的位置是`stride_h`和`stride_w`, 取得最大值后存放在输出特征图中对应的位置中，输出存放的位置为输出特征图`outut_channel`的`(int(r/stride_h),int(c/stride_w))`的位置中。**这部分可能描述地比较晦涩，请结合视频一起食用。**

## Max Pooling Layer的其他部分

```c++
MaxPoolingLayer::MaxPoolingLayer(const std::shared_ptr<Operator> &op) : Layer("maxpooling") {
  CHECK(op->op_type_ == OpType::kOperatorMaxPooling) << "Operator has a wrong type: " << int(op->op_type_);
  MaxPoolingOp *max_pooling_op = dynamic_cast<MaxPoolingOp *>(op.get());

  CHECK(max_pooling_op != nullptr) << "MaxPooling operator is empty";
  this->op_ = std::make_unique<MaxPoolingOp>(*max_pooling_op);
}

LayerRegistererWrapper kMaxPoolingLayer(OpType::kOperatorMaxPooling, MaxPoolingLayer::CreateInstance);
```

以上的步骤完成了`Max Pooling`层的注册, 具体流程已经在第五节中讲过。`MaxPoolingLayer::MaxPoolingLayer`初始化部分根据传入的`op`对`this->op_`进行赋值，`this->op_`中保存了``stride``,`padding`,`pooling`等计算时需要的属性信息。

## 单元测试

```C++
TEST(test_layer, forward_maxpooling1) {
  using namespace kuiper_infer;
  uint32_t stride_h = 1;
  uint32_t stride_w = 1;
  uint32_t padding_h = 0;
  uint32_t padding_w = 0;
  uint32_t pooling_h = 2;
  uint32_t pooling_w = 2;

  std::shared_ptr<Operator>
      max_op = std::make_shared<MaxPoolingOp>(pooling_h, pooling_w, stride_h, stride_w, padding_h, padding_w);
  std::shared_ptr<Layer> max_layer = LayerRegisterer::CreateLayer(max_op);
  CHECK(max_layer != nullptr);

  arma::fmat input_data = "0 1 2 ;"
                          "3 4 5 ;"
                          "6 7 8 ;";
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(2, input_data.n_rows, input_data.n_cols);
  input->at(0) = input_data;
  input->at(1) = input_data;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  max_layer->Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);
  const auto &output = outputs.at(0);
  LOG(INFO) << "\n" << output->data();
  ASSERT_EQ(output->rows(), 2);
  ASSERT_EQ(output->cols(), 2);

  ASSERT_EQ(output->at(0, 0, 0), 4);
  ASSERT_EQ(output->at(0, 0, 1), 5);
  ASSERT_EQ(output->at(0, 1, 0), 7);
  ASSERT_EQ(output->at(0, 1, 1), 8);

  ASSERT_EQ(output->at(1, 0, 0), 4);
  ASSERT_EQ(output->at(1, 0, 1), 5);
  ASSERT_EQ(output->at(1, 1, 0), 7);
  ASSERT_EQ(output->at(1, 1, 1), 8);
}
```

可以看到, 我们的输入为  `arma::fmat input_data ="0 1 2 ; 3 4 5 ;6 7 8; "` , 池化核的大小为2, 每次移动的步长`stride =1`,所以根据我们在第一节中的计算, 最后的输出特征图大小应该是2乘2大小, 池化得到的值分别为4 5 7 8.

![](https://picx.zhimg.com/80/v2-aa9f2de42e43d12206915c65ecc02d2d_1440w.png?source=d16d100b)