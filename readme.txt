========================================================================
   HỆ THỐNG DỊCH MÁY ANH → VIỆT (NEURAL MACHINE TRANSLATION)
========================================================================

Ngôn ngữ: Python 3.10+
Mô hình:  Transformer (From Scratch)
Tác giả:  Phùng Phúc Hậu & Phạm Trí Hùng

========================================================================

I. GIỚI THIỆU
-------------

Dự án xây dựng mô hình AI dịch thuật từ tiếng Anh sang tiếng Việt.
Hệ thống sử dụng kiến trúc Transformer hiện đại (tương tự Google Translate).

Hệ thống hỗ trợ hai chế độ dịch:
	1. Greedy Decoding  : Dịch nhanh
	2. Beam Search      : Dịch chính xác hơn (tìm kiếm theo chùm)

Ngoài giao diện console, hệ thống có mở rộng thành web dịch với giao diện đẹp và dễ sử dụng.

------------------------------------------------------------------------

II. CẤU TRÚC THƯ MỤC
-------------------

[Source Code & Cài đặt]
  + model.py           : Kiến trúc mạng Transformer (lõi hệ thống)
  + train.py           : Huấn luyện mô hình
  + translate.py       : Dịch thử trên Console
  + web_app.py         : Server Web (FastAPI)
  + cai_thu_vien.bat   : Cài thư viện tự động (Windows)
  + go_thu_vien.bat    : Gỡ thư viện tự động (Windows)
  + requirements.txt   : Danh sách thư viện cần thiết

[Frontend Web]
  + templates/
      - index.html     : Giao diện web
  + static/
      - style.css      : Giao diện (CSS)
      - app.js         : Xử lý logic web

[Dữ liệu sinh ra - Chỉ có sau khi Train]
  + vocab.pkl          : Từ điển (token ↔ id)
  + transformer.pth    : Trọng số mô hình
  + readme.txt         : File hướng dẫn này

------------------------------------------------------------------------

III. YÊU CẦU HỆ THỐNG
--------------------

1. Phần mềm:
   - Python phiên bản 3.10 hoặc 3.11
   - Link tải: https://www.python.org/downloads/

   QUAN TRỌNG:
  	 Khi cài Python, BẮT BUỘC tick chọn: "Add Python to PATH"

2. Phần cứng:
   - Khuyến nghị: Có GPU NVIDIA để train nhanh hơn
   - Tối thiểu: CPU vẫn chạy được (train lâu hơn)

------------------------------------------------------------------------

IV. CÀI ĐẶT THƯ VIỆN
-------------------

CÁCH 1: TỰ ĐỘNG (KHUYẾN NGHỊ - WINDOWS)
--------------------------------------
1. Tìm file: cai_thu_vien.bat
2. Click đúp chuột để chạy
3. Chờ đến khi hiện thông báo: "CAI DAT THANH CONG"

(Lưu ý: Máy cần có kết nối Internet)

CÁCH 2: THỦ CÔNG
----------------
Mở CMD hoặc Terminal tại thư mục dự án và chạy:

  pip install -r requirements.txt

------------------------------------------------------------------------

V. HUẤN LUYỆN MÔ HÌNH (TRAINING)
-------------------------------

Chạy lệnh:

  python train.py

Kết quả sau khi train xong:
  - vocab.pkl
  - transformer.pth

Lưu ý:
- Thời gian train phụ thuộc vào cấu hình máy
- Nếu thay đổi tham số model phải train lại từ đầu

------------------------------------------------------------------------

VI. DỊCH BẰNG CONSOLE
--------------------

Chạy lệnh:

  python translate.py

Cách sử dụng:
- Nhập câu tiếng Anh bất kỳ
- Hệ thống trả về bản dịch tiếng Việt
- Gõ 'q' để thoát chương trình

------------------------------------------------------------------------

VII. CHẠY WEB DỊCH 
---------------------------------

Tại thư mục dự án, chạy:

  uvicorn web_app:app --reload --host 0.0.0.0 --port 8000

3. Mở trình duyệt
-----------------
Truy cập địa chỉ:

  http://localhost:8000

Web cho phép:
- Nhập tiếng Anh
- Dịch sang tiếng Việt
- Copy kết quả
- Lưu lịch sử dịch

------------------------------------------------------------------------

VIII. KHẮC PHỤC SỰ CỐ (TROUBLESHOOTING)
--------------------------------------

LỖI 1: Click cai_thu_vien.bat nhưng tắt ngay
Nguyên nhân:
- Chưa cài Python
- Chưa Add Python to PATH

Khắc phục:
- Cài lại Python
- Nhớ tick "Add Python to PATH"

------------------------------------------------

LỖI 2: ModuleNotFoundError: No module named 'torch'
Nguyên nhân:
- Chưa cài thư viện hoặc cài bị lỗi

Khắc phục:
- Chạy lại cai_thu_vien.bat
- Hoặc chạy: pip install -r requirements.txt

------------------------------------------------

LỖI 3: FileNotFoundError: vocab.pkl hoặc transformer.pth
Nguyên nhân:
- Chưa train mà đã chạy dịch

Khắc phục:
- Chạy: python train.py

------------------------------------------------

LỖI 4: Lỗi load model (RuntimeError)
Nguyên nhân:
- Cấu hình model trong translate.py không khớp train.py

Khắc phục:
- Đảm bảo các tham số sau giống hệt khi train:
  D_MODEL
  NUM_HEADS
  NUM_LAYERS
  D_FF
  DROPOUT

------------------------------------------------------------------------

IX. GHI CHÚ
----------

- Beam Search cho kết quả tự nhiên hơn nhưng chậm hơn Greedy
- Muốn tăng chất lượng dịch:
  + Tăng dữ liệu train
  + Tăng số epoch
  + Tăng kích thước model (cần train lại)

========================================================================
                              HẾT
========================================================================
