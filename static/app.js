const $ = (id) => document.getElementById(id);

const src = $("src");
const tgt = $("tgt");
const btnTranslate = $("btnTranslate");
const btnClear = $("btnClear");
const btnCopy = $("btnCopy");
const status = $("status");
const hint = $("hint");
const meta = $("meta");
const historyList = $("historyList");
const btnClearHistory = $("btnClearHistory");

const LS_KEY = "envi_history_v1";

function setStatus(msg, isError=false){
  status.textContent = msg || "";
  status.style.opacity = msg ? 1 : 0;
  status.style.color = isError ? "rgba(255,200,200,.9)" : "rgba(255,255,255,.65)";
}

function saveHistory(item){
  const arr = JSON.parse(localStorage.getItem(LS_KEY) || "[]");
  arr.unshift(item);
  localStorage.setItem(LS_KEY, JSON.stringify(arr.slice(0, 20)));
  renderHistory();
}

function renderHistory(){
  const arr = JSON.parse(localStorage.getItem(LS_KEY) || "[]");
  historyList.innerHTML = "";
  if(arr.length === 0){
    historyList.innerHTML = `<div class="item"><div class="k">Trống</div><div class="v">Chưa có bản dịch nào.</div></div>`;
    return;
  }
  arr.forEach((it, idx) => {
    const div = document.createElement("div");
    div.className = "item";
    div.innerHTML = `
      <div class="k">#${idx+1} • ${new Date(it.t).toLocaleString()}</div>
      <div class="k">English</div>
      <div class="v">${escapeHtml(it.en)}</div>
      <div class="k" style="margin-top:8px">Vietnamese</div>
      <div class="v">${escapeHtml(it.vi)}</div>
    `;
    div.addEventListener("click", () => {
      src.value = it.en;
      tgt.value = it.vi;
      setStatus("Đã nạp lại từ lịch sử.");
    });
    historyList.appendChild(div);
  });
}

function escapeHtml(s){
  return (s || "").replace(/[&<>"']/g, (c) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"
  }[c]));
}

async function doTranslate(){
  const text = src.value.trim();
  if(!text){
    hint.textContent = "Nhập tiếng Anh trước nhé.";
    setStatus("");
    return;
  }
  hint.textContent = "";
  setStatus("Đang dịch…");
  btnTranslate.disabled = true;

  try{
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({text})
    });
    const data = await res.json();
    if(!data.ok){
      setStatus(data.error || "Có lỗi xảy ra.", true);
      return;
    }
    tgt.value = data.translation || "";
    setStatus("Xong!");
    saveHistory({ t: Date.now(), en: text, vi: tgt.value });
  }catch(e){
    setStatus("Không gọi được API. Kiểm tra server đang chạy chưa.", true);
  }finally{
    btnTranslate.disabled = false;
  }
}

btnTranslate.addEventListener("click", doTranslate);

btnClear.addEventListener("click", () => {
  src.value = "";
  tgt.value = "";
  setStatus("");
  hint.textContent = "";
  src.focus();
});

btnCopy.addEventListener("click", async () => {
  const text = tgt.value.trim();
  if(!text) return setStatus("Chưa có kết quả để copy.", true);
  await navigator.clipboard.writeText(text);
  setStatus("Đã copy vào clipboard!");
});

btnClearHistory.addEventListener("click", () => {
  localStorage.removeItem(LS_KEY);
  renderHistory();
  setStatus("Đã xoá lịch sử.");
});

document.addEventListener("keydown", (e) => {
  if((e.ctrlKey || e.metaKey) && e.key === "Enter"){
    doTranslate();
  }
});

renderHistory();
