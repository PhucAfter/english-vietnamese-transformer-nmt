@echo off
title GO BO MOI TRUONG (UNINSTALL)
color 0E

echo ========================================================
echo        TU DONG GO BO THU VIEN (CLEANUP)
echo ========================================================
echo.

:: Danh sách các thư viện cần gỡ
set LIST_PACKAGES=torch torchvision torchaudio datasets sacrebleu pyvi tqdm

echo Dang tien hanh go bo cac thu vien...
echo.

:: Lệnh gỡ cài đặt (-y nghĩa là tự động chọn Yes)
pip uninstall -y %LIST_PACKAGES%

:: --- SỬA LỖI LOGIC TẠI ĐÂY ---
:: Nếu có lỗi (errorlevel khác 0) thì nhảy ngay đến nhãn :CO_LOI
if %errorlevel% neq 0 goto :CO_LOI

:: Nếu không có lỗi, chạy tiếp phần này (Thành công)
:THANH_CONG
echo.
echo ========================================================
echo             DA GO BO THANH CONG! (CLEANED)
echo ========================================================
echo Moi truong Python da sach se nhu ban dau.
:: Quan trọng: Nhảy đến cuối cùng, bỏ qua phần lỗi
goto :KET_THUC

:CO_LOI
echo.
color 0C
echo ========================================================
echo             CO LOI XAY RA!
echo ========================================================
echo Co the thu vien chua tung duoc cai hoac ban khong co quyen Admin.

:KET_THUC
echo.
pause