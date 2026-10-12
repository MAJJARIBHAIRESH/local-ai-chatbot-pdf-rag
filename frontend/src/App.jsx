
import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

import "highlight.js/styles/github-dark.css";
import "./App.css";


const API_BASE = "http://127.0.0.1:8001";
const MAX_HISTORY_MESSAGES = 8;
const STORAGE_KEY = "ollama-chat-history-v1";

function createChat() {
  return {
    id: crypto.randomUUID(),
    title: "New Chat",
    messages: [],
    updatedAt: Date.now(),
  };
}

function loadChats() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    const chats = saved ? JSON.parse(saved) : [];
    return Array.isArray(chats) ? chats : [];
  } catch {
    return [];
  }
}

function extractText(node) {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }
  if (Array.isArray(node)) return node.map(extractText).join("");
  return extractText(node.props?.children);
}

async function readError(response) {
  let detail = `HTTP ${response.status}`;
  try {
    const data = await response.json();
    detail = data.detail || detail;
  } catch {
    // Keep the HTTP status when the response isn't JSON.
  }
  return detail;
}

function App() {
  const initialChats = loadChats();

  const [chats, setChats] = useState(initialChats);
  const [currentChatId, setCurrentChatId] = useState(
    () => initialChats[0]?.id ?? null
  );
  const [messages, setMessages] = useState(
    () => initialChats[0]?.messages ?? []
  );

  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copiedText, setCopiedText] = useState("");

  // PDF state
  const [pdfFile, setPdfFile] = useState(null);
  const [documentId, setDocumentId] = useState("");
  const [pdfFilename, setPdfFilename] = useState("");
  const [uploadingPdf, setUploadingPdf] = useState(false);
  const [pdfMode, setPdfMode] = useState(false);

  const abortRef = useRef(null);
  const chatEndRef = useRef(null);
  const loadingRef = useRef(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
    } catch (err) {
      console.error("Could not save chat history:", err);
    }
  }, [chats]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({
      behavior: loading ? "auto" : "smooth",
      block: "end",
    });
  }, [messages, loading]);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  function saveChatMessages(chatId, nextMessages) {
    setChats((previous) =>
      previous.map((chat) =>
        chat.id === chatId
          ? {
              ...chat,
              messages: nextMessages,
              updatedAt: Date.now(),
            }
          : chat
      )
    );
  }

  function updateAssistant(text, chatId) {
    setMessages((previous) => {
      const updated = [...previous];
      const last = updated.length - 1;

      if (last >= 0 && updated[last].role === "assistant") {
        updated[last] = { ...updated[last], text };
      }

      return updated;
    });

    setChats((previous) =>
      previous.map((chat) => {
        if (chat.id !== chatId) return chat;

        const updated = [...(chat.messages || [])];
        const last = updated.length - 1;

        if (last >= 0 && updated[last].role === "assistant") {
          updated[last] = { ...updated[last], text };
        }

        return {
          ...chat,
          messages: updated,
          updatedAt: Date.now(),
        };
      })
    );
  }

  function appendConversation(chatId, userText, titleFromQuestion = false) {
    const userMessage = { role: "user", text: userText };
    const assistantMessage = { role: "assistant", text: "" };

    setMessages((previous) => [
      ...previous,
      userMessage,
      assistantMessage,
    ]);

    setChats((previous) => {
      const existing = previous.find((chat) => chat.id === chatId);

      if (!existing) {
        return [
          {
            id: chatId,
            title: titleFromQuestion
              ? userText.slice(0, 45) +
                (userText.length > 45 ? "…" : "")
              : "New Chat",
            messages: [userMessage, assistantMessage],
            updatedAt: Date.now(),
          },
          ...previous,
        ];
      }

      const nextMessages = [
        ...(existing.messages || []),
        userMessage,
        assistantMessage,
      ];

      return [
        {
          ...existing,
          title:
            titleFromQuestion && existing.messages.length === 0
              ? userText.slice(0, 45) +
                (userText.length > 45 ? "…" : "")
              : existing.title,
          messages: nextMessages,
          updatedAt: Date.now(),
        },
        ...previous.filter((chat) => chat.id !== chatId),
      ];
    });
  }

  async function copyText(text, key) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedText(key);

      window.setTimeout(() => {
        setCopiedText((old) => (old === key ? "" : old));
      }, 1500);
    } catch {
      setError("Copy failed. Check browser clipboard permissions.");
    }
  }

  async function streamAnswer(url, body, chatId, signal) {
    let answer = "";
    let reader = null;

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      });

      if (!response.ok) {
        throw new Error(await readError(response));
      }

      if (!response.body) {
        throw new Error("Streaming is not supported by this browser.");
      }

      reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const result = await reader.read();
        if (result.done) break;

        if (result.value) {
          answer += decoder.decode(result.value, { stream: true });
          updateAssistant(answer, chatId);
        }
      }

      answer += decoder.decode();
      updateAssistant(answer, chatId);

      if (!answer.trim()) {
        throw new Error("The model returned an empty response. Try again.");
      }
    } catch (err) {
      if (err.name === "AbortError") {
        updateAssistant(
          answer
            ? `${answer}\n\n*Response stopped by you.*`
            : "Response stopped.",
          chatId
        );
      } else {
        console.error("Chat request failed:", err);

        const detail =
          err instanceof TypeError
            ? "Cannot reach FastAPI. Check that the backend is running at 127.0.0.1:8000."
            : err.message || "Something went wrong.";

        setError(detail);

        updateAssistant(
          answer
            ? `${answer}\n\n**Error:** ${detail}`
            : `**Error:** ${detail}`,
          chatId
        );
      }
    } finally {
      if (reader) {
        try {
          reader.releaseLock();
        } catch {
          // Reader may already be released.
        }
      }
    }
  }

  async function sendMessage(event) {
    event.preventDefault();

    const question = message.trim();
    if (!question || loadingRef.current) return;

    if (pdfMode && !documentId) {
      setError("Upload a PDF successfully before asking PDF questions.");
      return;
    }

    let chatId = currentChatId;
    const existingMessages = messages;

    if (!chatId) {
      const newChat = createChat();
      chatId = newChat.id;
      setCurrentChatId(chatId);
      setChats((previous) => [newChat, ...previous]);
    }

    const history = [
      ...existingMessages,
      { role: "user", text: question },
    ];

    const controller = new AbortController();
    abortRef.current = controller;
    loadingRef.current = true;

    setLoading(true);
    setError("");
    setMessage("");

    const titleFromQuestion = existingMessages.length === 0;
    appendConversation(chatId, question, titleFromQuestion);

    try {
      if (pdfMode) {
        await streamAnswer(
          `${API_BASE}/pdf/chat`,
          {
            document_id: documentId,
            question,
          },
          chatId,
          controller.signal
        );
      } else {
        const recentHistory = history.slice(-MAX_HISTORY_MESSAGES);

        await streamAnswer(
          `${API_BASE}/chat`,
          {
            messages: recentHistory.map((item) => ({
              role: item.role,
              content: item.text,
            })),
          },
          chatId,
          controller.signal
        );
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }

      loadingRef.current = false;
      setLoading(false);
    }
  }

  async function uploadPdf() {
    if (loadingRef.current || uploadingPdf) return;

    if (!pdfFile) {
      setError("Choose a PDF file first.");
      return;
    }

    if (!pdfFile.name.toLowerCase().endsWith(".pdf")) {
      setError("Please select a PDF file.");
      return;
    }

    if (pdfFile.size > 15 * 1024 * 1024) {
      setError("The PDF must be 15 MB or smaller.");
      return;
    }

    setUploadingPdf(true);
    setError("");
    setDocumentId("");
    setPdfFilename("");

    try {
      const formData = new FormData();
      formData.append("file", pdfFile);

      const response = await fetch(`${API_BASE}/pdf/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await readError(response));
      }

      const data = await response.json();

      setDocumentId(data.document_id);
      setPdfFilename(data.filename || pdfFile.name);
      setPdfMode(true);
      setError("");

      console.log("PDF uploaded successfully:", data);
    } catch (err) {
      console.error("PDF upload failed:", err);

      setError(
        err instanceof TypeError
          ? "Cannot reach the backend. Check that FastAPI is running."
          : err.message || "PDF upload failed."
      );
    } finally {
      setUploadingPdf(false);
    }
  }

  function removePdf() {
    setPdfFile(null);
    setDocumentId("");
    setPdfFilename("");
    setPdfMode(false);
    setError("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function startNewChat() {
    if (loadingRef.current) return;

    const newChat = createChat();
    setChats((previous) => [newChat, ...previous]);
    setCurrentChatId(newChat.id);
    setMessages([]);
    setMessage("");
    setError("");
  }

  function openChat(chat) {
    if (loadingRef.current) return;

    setCurrentChatId(chat.id);
    setMessages(chat.messages || []);
    setMessage("");
    setError("");
  }

  function deleteChat(chatId) {
    if (loadingRef.current) return;

    const remaining = chats.filter((chat) => chat.id !== chatId);
    setChats(remaining);

    if (chatId === currentChatId) {
      if (remaining.length > 0) {
        setCurrentChatId(remaining[0].id);
        setMessages(remaining[0].messages || []);
      } else {
        setCurrentChatId(null);
        setMessages([]);
      }
    }
  }

  return (
    <main className="app">
      <header className="header">
        <div className="logo">AI</div>

        <div className="header-info">
          <h1>AI Chat Assistant</h1>
          <p>Powered by Ollama · Llama 3.2</p>
        </div>

        <button
          type="button"
          className="new-chat"
          onClick={startNewChat}
          disabled={loading || uploadingPdf}
        >
          + New Chat
        </button>
      </header>

      <div className="chat-layout">
        <aside className="history-panel">
          <h2>Chat History</h2>

          {chats.length === 0 ? (
            <p className="history-empty">No saved conversations yet.</p>
          ) : (
            <ul className="history-list">
              {chats.map((chat) => (
                <li
                  key={chat.id}
                  className={
                    chat.id === currentChatId
                      ? "history-item active"
                      : "history-item"
                  }
                >
                  <button
                    type="button"
                    className="history-open"
                    onClick={() => openChat(chat)}
                    disabled={loading || uploadingPdf}
                    title={chat.title}
                  >
                    <span>{chat.title}</span>
                    <small>{chat.messages.length} messages</small>
                  </button>

                  <button
                    type="button"
                    className="history-delete"
                    onClick={() => deleteChat(chat.id)}
                    disabled={loading || uploadingPdf}
                    aria-label={`Delete ${chat.title}`}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <section className="chat" aria-live="polite" aria-busy={loading}>
          {messages.length === 0 && (
            <div className="welcome">
              <h2>How can I help you today?</h2>
              <p>Your local AI assistant, running on your computer.</p>

              <button
                type="button"
                className="suggestion"
                onClick={() =>
                  setMessage("Explain machine learning in simple English.")
                }
              >
                Explain machine learning →
              </button>
            </div>
          )}

          {messages.map((item, index) => (
            <div
              className={`message ${item.role}`}
              key={`${currentChatId}-${index}`}
            >
              <div className="message-heading">
                <strong className="message-author">
                  {item.role === "user" ? "You" : "AI Assistant"}
                </strong>

                {item.role === "assistant" && item.text && (
                  <button
                    type="button"
                    className="copy-response"
                    onClick={() =>
                      copyText(item.text, `response-${index}`)
                    }
                  >
                    {copiedText === `response-${index}`
                      ? "Copied!"
                      : "Copy response"}
                  </button>
                )}
              </div>

              <div
                className={`message-text ${
                  item.role === "assistant"
                    ? "markdown-body"
                    : "user-text"
                }`}
              >
                {item.text ? (
                  item.role === "assistant" ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                      components={{
                        a: ({ ...props }) => (
                          <a
                            {...props}
                            target="_blank"
                            rel="noopener noreferrer"
                          />
                        ),
                        pre: ({ children, ...props }) => {
                          const code = extractText(children);
                          const key = `code-${index}-${code}`;

                          return (
                            <div className="code-block-wrapper">
                              <button
                                type="button"
                                className="copy-code"
                                onClick={() => copyText(code, key)}
                              >
                                {copiedText === key
                                  ? "Copied!"
                                  : "Copy code"}
                              </button>

                              <pre className="code-block" {...props}>
                                {children}
                              </pre>
                            </div>
                          );
                        },
                      }}
                    >
                      {item.text}
                    </ReactMarkdown>
                  ) : (
                    item.text
                  )
                ) : loading && item.role === "assistant" ? (
                  "Thinking…"
                ) : null}
              </div>
            </div>
          ))}

          <div ref={chatEndRef} />
        </section>
      </div>

      {/* PDF upload panel */}
      <section className="pdf-panel">
        <div className="pdf-panel-heading">
          <strong>📄 Ask questions about a PDF</strong>
          <span className="pdf-help">
            PDF only · Maximum 15 MB
          </span>
        </div>

        <div className="pdf-controls">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,application/pdf"
            aria-label="Choose a PDF file"
            disabled={uploadingPdf || loading}
            onChange={(event) => {
              setPdfFile(event.target.files?.[0] || null);
              setError("");
            }}
          />

          <button
            type="button"
            onClick={uploadPdf}
            disabled={!pdfFile || uploadingPdf || loading}
          >
            {uploadingPdf ? "Processing PDF…" : "Upload PDF"}
          </button>

          {documentId && (
            <button
              type="button"
              className="pdf-remove"
              onClick={removePdf}
              disabled={uploadingPdf || loading}
            >
              Remove PDF
            </button>
          )}
        </div>

        {documentId && (
          <p className="pdf-success" role="status">
            ✓ Ready: {pdfFilename}. Switch to PDF mode to ask questions.
          </p>
        )}

        <label className="pdf-mode-toggle">
          <input
            type="checkbox"
            checked={pdfMode}
            onChange={(event) => {
              const enabled = event.target.checked;

              if (enabled && !documentId) {
                setError("Upload a PDF successfully before enabling PDF mode.");
                setPdfMode(false);
                return;
              }

              setError("");
              setPdfMode(enabled);
            }}
            disabled={!documentId || loading || uploadingPdf}
          />
          Ask questions using the uploaded PDF
        </label>
      </section>

      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}

      <form className="input-area" onSubmit={sendMessage}>
        <input
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          placeholder={
            pdfMode
              ? "Ask a question about your PDF..."
              : "Ask me anything..."
          }
          aria-label="Your question"
          disabled={loading || uploadingPdf}
        />

        {loading ? (
          <button
            type="button"
            className="stop-button"
            onClick={() => abortRef.current?.abort()}
          >
            Stop ■
          </button>
        ) : (
          <button
            type="submit"
            disabled={!message.trim() || uploadingPdf}
          >
            Send ↑
          </button>
        )}
      </form>

      <footer>Local AI · Your questions stay on your computer</footer>
    </main>
  );
}

export default App;
