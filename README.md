# ETT Temperature Forecasting Web Application

Dự báo dầu biến áp (Oil Temperature — OT) trên dataset **ETTm1** thông qua giao diện Web trực quan. Ứng dụng cung cấp các kết quả đánh giá, biểu đồ so sánh và giao diện cho phép dự báo trực tiếp với các mô hình deep learning.

---

## Cấu trúc project

```text
ETTForecasting/
├── app.py                 ← File chạy chính của ứng dụng Flask (Web server)
├── data/                  ← Chứa dữ liệu đầu vào (VD: ETTm1.csv)
├── models/                ← Nơi lưu trọng số mô hình đã được huấn luyện (.pth)
├── src/                   ← Mã nguồn Python xử lý thuật toán
│   ├── preprocess.py      ← Xử lý dữ liệu, định nghĩa Data Pipeline
│   ├── models.py          ← Định nghĩa kiến trúc các mô hình
│   ├── metrics.py         ← Hàm tính toán các chỉ số đánh giá (MSE, MAE...)
│   ├── inference.py       ← Xử lý dự báo trên dữ liệu test
│   └── plots.py           ← Vẽ biểu đồ đánh giá
├── static/                ← CSS, JS, hình ảnh cho web tĩnh
├── templates/             ← HTML templates cho Flask (Giao diện web)
├── artifacts/             ← Lưu trữ các kết quả, metrics được tính toán sẵn
└── requirements.txt       ← Danh sách các thư viện phụ thuộc
```

---

## Cài đặt thư viện

Mở terminal (ví dụ: Command Prompt, PowerShell, hoặc Terminal trong VSCode) tại thư mục gốc của project, và chạy lệnh sau để cài đặt các thư viện yêu cầu:

```bash
pip install -r requirements.txt
```

*(Lưu ý: Bạn nên sử dụng môi trường ảo như venv hoặc conda trước khi cài đặt).*

---

## Hướng dẫn chạy Web Application

Để khởi động ứng dụng Web (Flask Dashboard & API), bạn chạy lệnh sau tại thư mục gốc:

```bash
python app.py
```

Khi Terminal hiển thị thông báo ứng dụng đã chạy, server sẽ lắng nghe ở mọi địa chỉ (`0.0.0.0`) trên cổng `5000`.

Truy cập trang web thông qua trình duyệt bằng đường dẫn sau:
👉 **[http://localhost:5000](http://localhost:5000)** hoặc **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

*(Lưu ý: Để có thể chia sẻ ứng dụng cho các thiết bị khác trong cùng mạng LAN truy cập, bạn có thể gửi cho họ địa chỉ IP mạng của máy bạn + cổng 5000. Ví dụ: `http://192.168.1.xxx:5000`)*.

---

## Các chức năng chính (Routes) trên Web

- **Dashboard (`/`):** Xem bảng xếp hạng tổng quan và các số liệu so sánh các mô hình.
- **Pipeline (`/pipeline`):** Tổng quan về dữ liệu, danh sách các features và tham số chuẩn hóa.
- **Models (`/models`):** Xem thông tin và cấu trúc của từng mô hình dự đoán.
- **Results (`/results`):** Biểu đồ kết quả trực quan, sự khác biệt giữa dự đoán và thực tế.
- **Visualize (`/visualize`):** Cho phép tương tác, chọn một mô hình cụ thể và mốc thời gian để vẽ đồ thị kết quả dự đoán so với đường giá thực ngay trên web.
- **Weights (`/weights`):** Hiển thị thống kê về trọng số mạng nơ-ron và layers.
