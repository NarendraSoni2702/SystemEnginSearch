import threading
import tkinter as tk

from tkinter import ttk, messagebox

import rag_engine


# ============================================================
# COLORS
# ============================================================

BG_COLOR = "#F5F7FA"

SIDEBAR_COLOR = "#1F2937"

HEADER_COLOR = "#FFFFFF"

CHAT_COLOR = "#FFFFFF"

USER_COLOR = "#DCF8C6"

AI_COLOR = "#F1F3F5"

TEXT_COLOR = "#111827"

MUTED_COLOR = "#6B7280"

ACCENT_COLOR = "#2563EB"

BORDER_COLOR = "#E5E7EB"


# ============================================================
# APPLICATION
# ============================================================

class LocalAIApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Local AI File Assistant"
        )

        self.root.geometry(
            "1050x720"
        )

        self.root.minsize(
            850,
            600
        )

        self.root.configure(
            bg=BG_COLOR
        )

        self.engine = None

        self.processing = False

        self.create_styles()

        self.create_ui()

        self.load_engine()


    # ========================================================
    # STYLES
    # ========================================================

    def create_styles(self):

        style = ttk.Style()

        try:

            style.theme_use(
                "clam"
            )

        except tk.TclError:

            pass

        style.configure(

            "Send.TButton",

            font=(
                "Segoe UI",
                10,
                "bold"
            ),

            padding=(
                18,
                10
            ),

            background=ACCENT_COLOR,

            foreground="white"
        )

        style.map(

            "Send.TButton",

            background=[
                (
                    "active",
                    "#1D4ED8"
                )
            ]
        )


    # ========================================================
    # MAIN UI
    # ========================================================

    def create_ui(self):

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = tk.Frame(

            self.root,

            bg=HEADER_COLOR,

            height=65,

            highlightbackground=BORDER_COLOR,

            highlightthickness=1
        )

        header.pack(
            fill="x"
        )

        header.pack_propagate(
            False
        )

        title_frame = tk.Frame(
            header,
            bg=HEADER_COLOR
        )

        title_frame.pack(
            side="left",
            padx=20
        )

        title = tk.Label(

            title_frame,

            text="🧠  Local AI File Assistant",

            font=(
                "Segoe UI",
                16,
                "bold"
            ),

            fg=TEXT_COLOR,

            bg=HEADER_COLOR
        )

        title.pack(
            side="left",
            pady=15
        )

        self.status_label = tk.Label(

            header,

            text="● Starting...",

            font=(
                "Segoe UI",
                10
            ),

            fg="#F59E0B",

            bg=HEADER_COLOR
        )

        self.status_label.pack(

            side="right",

            padx=20
        )


        # ----------------------------------------------------
        # MAIN AREA
        # ----------------------------------------------------

        main = tk.Frame(

            self.root,

            bg=BG_COLOR
        )

        main.pack(
            fill="both",
            expand=True
        )


        # ----------------------------------------------------
        # SIDEBAR
        # ----------------------------------------------------

        sidebar = tk.Frame(

            main,

            bg=SIDEBAR_COLOR,

            width=220
        )

        sidebar.pack(

            side="left",

            fill="y"
        )

        sidebar.pack_propagate(
            False
        )


        sidebar_title = tk.Label(

            sidebar,

            text="DOCUMENTS",

            font=(
                "Segoe UI",
                9,
                "bold"
            ),

            fg="#9CA3AF",

            bg=SIDEBAR_COLOR
        )

        sidebar_title.pack(

            anchor="w",

            padx=18,

            pady=18
        )


        self.files_label = tk.Label(

            sidebar,

            text="Loading...",

            font=(
                "Segoe UI",
                10
            ),

            fg="#E5E7EB",

            bg=SIDEBAR_COLOR,

            justify="left",

            anchor="nw"
        )

        self.files_label.pack(

            fill="x",

            padx=18
        )


        separator = tk.Frame(

            sidebar,

            bg="#374151",

            height=1
        )

        separator.pack(

            fill="x",

            padx=15,

            pady=20
        )


        info_title = tk.Label(

            sidebar,

            text="SYSTEM",

            font=(
                "Segoe UI",
                9,
                "bold"
            ),

            fg="#9CA3AF",

            bg=SIDEBAR_COLOR
        )

        info_title.pack(

            anchor="w",

            padx=18,

            pady=(0, 10)
        )


        self.model_label = tk.Label(

            sidebar,

            text="Model:\nLoading...",

            font=(
                "Segoe UI",
                9
            ),

            fg="#D1D5DB",

            bg=SIDEBAR_COLOR,

            justify="left",

            anchor="w"
        )

        self.model_label.pack(

            anchor="w",

            padx=18
        )


        # ----------------------------------------------------
        # CONTENT
        # ----------------------------------------------------

        content = tk.Frame(

            main,

            bg=BG_COLOR
        )

        content.pack(

            side="left",

            fill="both",

            expand=True
        )


        # ----------------------------------------------------
        # CHAT AREA
        # ----------------------------------------------------

        chat_container = tk.Frame(

            content,

            bg=CHAT_COLOR
        )

        chat_container.pack(

            fill="both",

            expand=True,

            padx=12,

            pady=(12, 6)
        )


        self.chat_text = tk.Text(

            chat_container,

            wrap="word",

            font=(
                "Segoe UI",
                11
            ),

            bg=CHAT_COLOR,

            fg=TEXT_COLOR,

            relief="flat",

            padx=20,

            pady=20,

            state="disabled"
        )

        scrollbar = ttk.Scrollbar(

            chat_container,

            orient="vertical",

            command=self.chat_text.yview
        )

        self.chat_text.configure(

            yscrollcommand=
            scrollbar.set
        )

        self.chat_text.pack(

            side="left",

            fill="both",

            expand=True
        )

        scrollbar.pack(

            side="right",

            fill="y"
        )


        # ----------------------------------------------------
        # TEXT TAGS
        # ----------------------------------------------------

        self.chat_text.tag_configure(

            "user_name",

            foreground=ACCENT_COLOR,

            font=(
                "Segoe UI",
                10,
                "bold"
            )
        )

        self.chat_text.tag_configure(

            "user_text",

            foreground=TEXT_COLOR,

            background=USER_COLOR,

            font=(
                "Segoe UI",
                11
            ),

            lmargin1=10,

            lmargin2=10
        )

        self.chat_text.tag_configure(

            "ai_name",

            foreground="#059669",

            font=(
                "Segoe UI",
                10,
                "bold"
            )
        )

        self.chat_text.tag_configure(

            "ai_text",

            foreground=TEXT_COLOR,

            background=AI_COLOR,

            font=(
                "Segoe UI",
                11
            ),

            lmargin1=10,

            lmargin2=10
        )

        self.chat_text.tag_configure(

            "system",

            foreground=MUTED_COLOR,

            font=(
                "Segoe UI",
                9
            )
        )

        self.chat_text.tag_configure(

            "source",

            foreground="#4B5563",

            font=(
                "Segoe UI",
                9
            )
        )


        # ----------------------------------------------------
        # INPUT AREA
        # ----------------------------------------------------

        input_outer = tk.Frame(

            content,

            bg=BG_COLOR
        )

        input_outer.pack(

            fill="x",

            padx=12,

            pady=(6, 12)
        )


        self.input_text = tk.Text(

            input_outer,

            height=3,

            font=(
                "Segoe UI",
                11
            ),

            wrap="word",

            relief="solid",

            bd=1,

            highlightthickness=1,

            highlightcolor=ACCENT_COLOR
        )

        self.input_text.pack(

            side="left",

            fill="both",

            expand=True
        )


        button_frame = tk.Frame(

            input_outer,

            bg=BG_COLOR
        )

        button_frame.pack(

            side="right",

            padx=(8, 0)
        )


        self.send_button = ttk.Button(

            button_frame,

            text="Send",

            style="Send.TButton",

            command=self.send_question
        )

        self.send_button.pack()


        clear_button = ttk.Button(

            button_frame,

            text="Clear",

            command=self.clear_chat
        )

        clear_button.pack(

            pady=(8, 0)
        )


        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        footer = tk.Frame(

            content,

            bg=HEADER_COLOR,

            height=30,

            highlightbackground=BORDER_COLOR,

            highlightthickness=1
        )

        footer.pack(

            fill="x"
        )

        footer.pack_propagate(
            False
        )


        self.footer_label = tk.Label(

            footer,

            text="Initializing...",

            font=(
                "Segoe UI",
                9
            ),

            fg=MUTED_COLOR,

            bg=HEADER_COLOR
        )

        self.footer_label.pack(

            side="left",

            padx=15
        )


        # ----------------------------------------------------
        # KEYBOARD SHORTCUT
        # ----------------------------------------------------

        self.input_text.bind(

            "<Control-Return>",

            self.send_question_event
        )

        self.input_text.bind(

            "<Return>",

            self.handle_enter
        )


    # ========================================================
    # LOAD ENGINE
    # ========================================================

    def load_engine(self):

        self.set_status(
            "● Loading...",
            "#F59E0B"
        )

        self.add_system_message(
            "Loading Ollama and document index..."
        )

        thread = threading.Thread(

            target=self._load_engine_worker,

            daemon=True
        )

        thread.start()


    def _load_engine_worker(self):

        try:

            def progress(message):

                self.root.after(

                    0,

                    lambda:
                    self.set_loading_message(
                        message
                    )
                )

            engine = rag_engine.initialize(
                progress
            )

            self.root.after(

                0,

                lambda:
                self.engine_ready(
                    engine
                )
            )

        except Exception as error:

            self.root.after(

                0,

                lambda:
                self.engine_error(
                    error
                )
            )


    # ========================================================
    # ENGINE READY
    # ========================================================

    def engine_ready(
            self,
            engine
    ):

        self.engine = engine

        files = engine["documents"]

        chunks = engine["chunks"]

        self.set_status(
            "● Ready",
            "#10B981"
        )

        self.files_label.config(

            text=
            f"Files: {len(files)}\n\n"
            f"Chunks: {len(chunks)}\n\n"
            +
            "\n".join(
                f"• {doc['filename']}"
                for doc in files
            )
        )

        self.model_label.config(

            text=
            f"LLM:\n{rag_engine.LLM_MODEL}\n\n"
            f"Embedding:\n"
            f"{rag_engine.EMBEDDING_MODEL}"
        )

        self.footer_label.config(

            text=
            f"Files: {len(files)}    "
            f"Chunks: {len(chunks)}    "
            f"Model: {rag_engine.LLM_MODEL}"
        )

        self.add_system_message(

            "Document index ready. "
            "You can now ask questions about your files."
        )

        self.input_text.focus_set()


    # ========================================================
    # ENGINE ERROR
    # ========================================================

    def engine_error(
            self,
            error
    ):

        self.set_status(
            "● Error",
            "#EF4444"
        )

        self.add_system_message(

            f"ERROR:\n{error}"
        )

        messagebox.showerror(

            "Initialization Error",

            str(error)
        )


    # ========================================================
    # STATUS
    # ========================================================

    def set_status(
            self,
            text,
            color
    ):

        self.status_label.config(

            text=text,

            fg=color
        )


    def set_loading_message(
            self,
            message
    ):

        self.footer_label.config(
            text=message
        )


    # ========================================================
    # CHAT DISPLAY
    # ========================================================

    def add_system_message(
            self,
            message
    ):

        self.chat_text.config(
            state="normal"
        )

        self.chat_text.insert(

            "end",

            "\n"

            + message

            + "\n\n",

            "system"
        )

        self.chat_text.config(
            state="disabled"
        )

        self.chat_text.see(
            "end"
        )


    def add_user_message(
            self,
            message
    ):

        self.chat_text.config(
            state="normal"
        )

        self.chat_text.insert(

            "end",

            "\nYou\n",

            "user_name"
        )

        self.chat_text.insert(

            "end",

            message
            +
            "\n\n",

            "user_text"
        )

        self.chat_text.config(
            state="disabled"
        )

        self.chat_text.see(
            "end"
        )


    def add_ai_message(
            self,
            answer,
            sources=None,
            timing=None
    ):

        self.chat_text.config(
            state="normal"
        )

        self.chat_text.insert(

            "end",

            "\nAI\n",

            "ai_name"
        )

        self.chat_text.insert(

            "end",

            answer
            +
            "\n\n",

            "ai_text"
        )


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if sources:

            self.chat_text.insert(

                "end",

                "Sources\n",

                "source"
            )

            seen = set()

            for source in sources:

                filename = source["filename"]

                if filename in seen:

                    continue

                seen.add(filename)

                self.chat_text.insert(

                    "end",

                    f"• {filename} "
                    f"(similarity: "
                    f"{source['score']:.3f})\n",

                    "source"
                )

            self.chat_text.insert(

                "end",

                "\n",

                "source"
            )


        # ----------------------------------------------------
        # TIMING
        # ----------------------------------------------------

        if timing:

            self.chat_text.insert(

                "end",

                (
                    f"Search: "
                    f"{timing.get('search_time', 0):.2f}s   "
                    f"| AI: "
                    f"{timing.get('ai_time', 0):.2f}s   "
                    f"| Total: "
                    f"{timing.get('total_time', 0):.2f}s\n\n"
                ),

                "system"
            )


        self.chat_text.config(
            state="disabled"
        )

        self.chat_text.see(
            "end"
        )


    # ========================================================
    # SEND QUESTION
    # ========================================================

    def send_question_event(
            self,
            event
    ):

        self.send_question()

        return "break"


    def handle_enter(
            self,
            event
    ):

        # Normal Enter = new line
        # Ctrl+Enter = send

        return None


    def send_question(self):

        if self.processing:

            return

        if self.engine is None:

            messagebox.showwarning(

                "Please wait",

                "The document index is still loading."
            )

            return

        question = self.input_text.get(

            "1.0",

            "end"
        ).strip()

        if not question:

            return

        if question.lower() == "exit":

            self.root.destroy()

            return

        self.input_text.delete(

            "1.0",

            "end"
        )

        self.add_user_message(
            question
        )

        self.processing = True

        self.send_button.config(
            state="disabled"
        )

        self.input_text.config(
            state="disabled"
        )

        self.set_status(
            "● Thinking...",
            "#F59E0B"
        )

        self.footer_label.config(

            text="Searching documents..."
        )

        thread = threading.Thread(

            target=self._question_worker,

            args=(question,),

            daemon=True
        )

        thread.start()


    # ========================================================
    # QUESTION WORKER
    # ========================================================

    def _question_worker(
            self,
            question
    ):

        try:

            result = rag_engine.ask_question(

                question,

                self.engine
            )

            self.root.after(

                0,

                lambda:
                self.question_complete(
                    result
                )
            )

        except Exception as error:

            self.root.after(

                0,

                lambda:
                self.question_error(
                    error
                )
            )


    # ========================================================
    # QUESTION COMPLETE
    # ========================================================

    def question_complete(
            self,
            result
    ):

        self.processing = False

        self.send_button.config(
            state="normal"
        )

        self.input_text.config(
            state="normal"
        )

        self.set_status(
            "● Ready",
            "#10B981"
        )

        self.footer_label.config(

            text=
            f"Model: {rag_engine.LLM_MODEL}    "
            f"Ready"
        )

        self.add_ai_message(

            result["answer"],

            result.get(
                "sources",
                []
            ),

            result
        )

        self.input_text.focus_set()


    # ========================================================
    # QUESTION ERROR
    # ========================================================

    def question_error(
            self,
            error
    ):

        self.processing = False

        self.send_button.config(
            state="normal"
        )

        self.input_text.config(
            state="normal"
        )

        self.set_status(
            "● Ready",
            "#10B981"
        )

        self.add_system_message(

            f"Error while processing question:\n"
            f"{error}"
        )

        self.input_text.focus_set()


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    def clear_chat(self):

        if self.processing:

            return

        self.chat_text.config(
            state="normal"
        )

        self.chat_text.delete(
            "1.0",
            "end"
        )

        self.chat_text.config(
            state="disabled"
        )

        self.add_system_message(

            "Chat cleared. "
            "Your document index is still available."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    root = tk.Tk()

    app = LocalAIApp(root)

    root.mainloop()


if __name__ == "__main__":

    main()