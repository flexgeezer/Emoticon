import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from config import BG, SURFACE, ACCENT, TEXT, MUTED, LARGEFONT, LABELFONT, BTN_BACK, BTN_SAVE, emoji_dict
from UI.base_page import BasePage
from UI.helpers import readarray, readcomment, savedata, get_file_name

EMOTIONS = ("Happy", "Sad", "Confused", "Angry")
EMO_COLORS = ("#a6e3a1", "#f38ba8", "#fab387", "#cba6f7")


class Report(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.add_back_button("Report History", 'ReportHistory')
        self._content = tk.Frame(self, bg=BG)
        self._content.place(relx=0, rely=0.09, relwidth=1.0, relheight=0.91)
        self._fig = None

    def load(self, file_path):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
        for w in self._content.winfo_children():
            w.destroy()

        array, times = readarray(file_path)
        comments = readcomment(file_path)

        # Left: title + graph
        left = tk.Frame(self._content, bg=BG)
        left.place(relx=0, rely=0, relwidth=0.62, relheight=1.0)

        tk.Label(left, text=get_file_name(file_path), font=LARGEFONT,
                 bg=BG, fg=ACCENT).pack(pady=(10, 4))
        self._fig = self._plot(left, array, times)

        # Right: emoji + stats + comments
        right = tk.Frame(self._content, bg=SURFACE)
        right.place(relx=0.63, rely=0, relwidth=0.37, relheight=1.0)

        if array:
            emo_index = int(np.argmax(np.mean(array, axis=0)))
            try:
                img = Image.open(emoji_dict[emo_index]).resize((110, 110))
                photo = ImageTk.PhotoImage(img)
                lbl = tk.Label(right, image=photo, bg=SURFACE)
                lbl.photo = photo
                lbl.pack(pady=(16, 6))
            except Exception:
                pass

            means = np.mean(array, axis=0)
            for emo, val, color in zip(EMOTIONS, means, EMO_COLORS):
                row = tk.Frame(right, bg=SURFACE)
                row.pack(fill='x', padx=16, pady=2)
                tk.Label(row, text=emo, font=LABELFONT, bg=SURFACE, fg=TEXT,
                         width=8, anchor='w').pack(side='left')
                tk.Label(row, text=f"{val:.1%}", font=LABELFONT,
                         bg=SURFACE, fg=color).pack(side='right')

        tk.Frame(right, bg=MUTED, height=1).pack(fill='x', padx=12, pady=(10, 6))
        tk.Label(right, text="Notes", font=("Verdana", 9, "bold"),
                 bg=SURFACE, fg=MUTED).pack(anchor='w', padx=14)

        comment_box = tk.Text(right, font=("Verdana", 9), relief="flat", wrap=tk.WORD,
                              bg=BG, fg=TEXT, insertbackground=TEXT,
                              highlightbackground=MUTED, highlightthickness=1)
        comment_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 6))
        comment_box.insert("1.0", comments)

        tk.Button(right, text="Save Notes", **BTN_SAVE,
                  command=lambda: savedata(file_path, comment_box.get("1.0", "end-1c"))).pack(pady=(0, 10))

    def _plot(self, parent, array, times, interval=3.0):
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(SURFACE)

            if array:
                x_vals, agg = self._aggregate(array, times, interval)
                agg_np = np.array(agg)
                for i, (emo, color) in enumerate(zip(EMOTIONS, EMO_COLORS)):
                    ax.plot(x_vals, agg_np[:, i], label=emo, color=color, linewidth=1.8)
                ax.set_xlabel("Time (s)" if times else "Frame", color=MUTED, fontsize=8)

            ax.set_title("Emotion over Time", color=TEXT, fontsize=10, pad=8)
            ax.set_ylabel("Confidence", color=MUTED, fontsize=8)
            ax.tick_params(colors=MUTED, labelsize=7)
            ax.spines[:].set_color(MUTED)
            legend = ax.legend(loc='upper right', fontsize=7, framealpha=0.3)
            plt.setp(legend.get_texts(), color=TEXT)
            ax.grid(True, alpha=0.15, color=MUTED)
            fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        return fig

    def _aggregate(self, array, times, interval):
        if not times:
            return list(range(len(array))), array
        num_buckets = int(max(times) / interval) + 1
        buckets = [[] for _ in range(num_buckets)]
        for row, t in zip(array, times):
            buckets[int(t / interval)].append(row)
        x_vals, agg = [], []
        for i, bucket in enumerate(buckets):
            if bucket:
                x_vals.append(round(i * interval, 1))
                agg.append(np.mean(bucket, axis=0).tolist())
        return x_vals, agg
