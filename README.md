# FIRST EXPERIMENT ON UET-AILAB 

## Thí nghiệm

Ném quả bóng lên cao theo hướng thẳng đứng, vận tốc ban đầu là $v_0$.

<img src="https://render.githubusercontent.com/render/math?math=y_t = v_0 t - \frac 1 2 gt^2">

Bài toán Học máy: Tìm mô hình mô tả chuyển động trên?

Giả thuyết: Chuyển động có thể biểu diễn bằng mô hình đa thức của $t$.

## Dữ liệu

Hàm sinh dữ liệu 
<img src="https://render.githubusercontent.com/render/math?math=y_t = v_0 t - \frac 1 2 gt^2 + \epsilon">

với
<img src="https://render.githubusercontent.com/render/math?math=\epsilon \sim \mathcal N(0, \sigma^2)">

File train: [`train.csv`](./data/train.csv)

File test: [`test.csv`](./data/test.csv)

## Lựa chọn tham số thí nghiệm
- Phương pháp tối ưu: *Gradient Descent*
- Siêu tham số:
    - **Batch size**: 1 (*Stochastic Gradient Descent*)
    - **Epochs**: 100
    - **Learning rate**: 0.01
- Tham số  đầu vào:

<img src="https://render.githubusercontent.com/render/math?math=d = 0,1,2,3,4,5,6,7,8,9,10 ">
 
## Số liệu từng lần chạy

<img src="https://render.githubusercontent.com/render/math?math=R^2 = 1 - \frac{\sum_{i=1}^n (y_i-f(t_i))^2}{\sum_{i=1}^n (y_i-\overline{y})^2}">

