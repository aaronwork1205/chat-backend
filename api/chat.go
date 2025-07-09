package api

import (
    "encoding/json"
    "io/ioutil"
    "net/http"
    "os/exec"
    "strings"

    "personal-chat/core"
)

type ChatRequest struct {
    Question string `json:"question"`
}

type ChatResponse struct {
    Answer string `json:"answer"`
}

func ChatHandler(w http.ResponseWriter, r *http.Request) {
    var req ChatRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "参数错误", http.StatusBadRequest)
        return
    }

    // 读取简历内容
    resume, err := ioutil.ReadFile("asset/Resume_Aaron_Wang.pdf")
    if err != nil {
        http.Error(w, "简历读取失败", http.StatusInternalServerError)
        return
    }

    // 这里假设你有一个 Python 脚本可以把 PDF 转为文本
    out, err := exec.Command("python3", "tokenizer/pdf2text.py", "asset/Resume_Aaron_Wang.pdf").Output()
    if err != nil {
        http.Error(w, "简历解析失败", http.StatusInternalServerError)
        return
    }
    resumeText := string(out)

    // 拼接 prompt
    prompt := "以下是我的简历内容：\n" + resumeText + "\n\nHR提问：" + req.Question + "\n请基于我的简历内容作答："

    // Tokenizer: 调用 Python 脚本获得 input_ids
    tokenOut, err := exec.Command("python3", "tokenizer/tokenize.py", prompt).Output()
    if err != nil {
        http.Error(w, "分词失败", http.StatusInternalServerError)
        return
    }
    // 假设输出为逗号分隔的数字
    tokenStrs := strings.Split(strings.TrimSpace(string(tokenOut)), ",")
    var inputIds []int64
    for _, s := range tokenStrs {
        if s == "" {
            continue
        }
        var id int64
        fmt.Sscanf(s, "%d", &id)
        inputIds = append(inputIds, id)
    }

    // ONNX 推理
    outputIds, err := core.RunONNXInference(inputIds)
    if err != nil {
        http.Error(w, "模型推理失败", http.StatusInternalServerError)
        return
    }

    // 解码：调用 Python 脚本
    idsStr := make([]string, len(outputIds))
    for i, id := range outputIds {
        idsStr[i] = fmt.Sprintf("%d", id)
    }
    decodedOut, err := exec.Command("python3", "tokenizer/decode.py", strings.Join(idsStr, ",")).Output()
    if err != nil {
        http.Error(w, "解码失败", http.StatusInternalServerError)
        return
    }
    answer := string(decodedOut)

    json.NewEncoder(w).Encode(ChatResponse{Answer: answer})
}