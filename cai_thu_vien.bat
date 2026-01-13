@echo off
title CAI DAT MOI TRUONG HE THONG DICH MAY
color 0A

echo ========================================================
echo         TU DONG CAI DAT THU VIEN (AUTO INSTALL)
echo ========================================================
echo.

:: 1. KIEM TRA PYTHON
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [LOI] May tinh chua cai Python.
    pause
    exit
)

:: 2. NANG CAP PIP
echo Dang kiem tra va nang cap PIP...
python -m pip install --upgrade pip
echo.

:: 3. CAI DAT THU VIEN
echo Dang cai dat cac thu vien...
echo.

:: --- Lệnh cài đặt chính ---
pip install -r requirements.txt

:: --- KIEM TRA LOI (LOGIC GOTO) ---
:: Nếu có lỗi (mã lỗi khác 0), nhảy ngay đến nhãn :CO_LOI
if %errorlevel% neq 0 goto :CO_LOI

:: --- PHẦN THÀNH CÔNG (Chỉ chạy khi không có lỗi) ---
:THANH_CONG
echo.
color 0A
echo ========================================================
echo             CAI DAT THANH CONG! (SUCCESS)
echo ========================================================
echo Moi thu da san sang.
echo Ban co the chay file 'train.py' hoac 'translate.py' ngay.
:: QUAN TRỌNG: Nhảy đến cuối cùng, bỏ qua phần lỗi
goto :KET_THUC

:: --- PHẦN LỖI (Chỉ chạy khi có lệnh nhảy vào đây) ---
:CO_LOI
echo.
color 0C
echo ========================================================
echo             CO LOI XAY RA! (ERROR)
echo ========================================================
echo Kiem tra lai ket noi Internet hoac phien ban Python.

:: --- KẾT THÚC ---
:KET_THUC
echo.
pause