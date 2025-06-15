
import os
import sys
import random
import functools
from typing import Generator, Callable, Any

import re

# python tkinter docks : "https://anzeljg.github.io/rin2/book2/2405/docs/tkinter/"
# sudo apt-get install python3-tk  Wtf linux? Idk how to add this dependency to the project config

import tkinter as tkint
import tkinter.ttk as tkintTtk
import tkinter.font as tkintFont
import tktooltip

tk = tkint

import gui_integration.utils as utils

log = utils.get_logger(__name__)


_sample_text = """
Whether we wanted it or not, we've stepped into a war with the
Cabal on Mars. So let's get to taking out their command, one by
one. Valus Ta'aurc. From what I can gather he commands the Siege
Dancers from an Imperial Land Tank outside of Rubicon. He's well 
protected, but with the right team, we can punch through those
defenses, take this beast out, and break their grip on Freehold.
""".lstrip('\n')

# todo-s:
# Reset button in tab 1 ???
# katram modelim savs threshold, to neaizmirsti
#


class _TextTagBindingProxy:
    """ For use with tktooltip.ToolTip """

    def __init__(self, ctx_text: tkint.Text, tag_name: str):
        self.ctx_text = ctx_text
        self.tag_name = tag_name

    def bind(self, sequence, func, add) -> str:
        return self.ctx_text.tag_bind(self.tag_name, sequence, func, add)

    def unbind(self, sequence, funcid) -> None:
        return self.ctx_text.tag_unbind(self.tag_name, sequence, funcid)


class Application:
    def __init__(self):
        window_extent = (800, 600)
        tk_root = tkint.Tk()
        tk_root.title("mordoria")
        tk_root.geometry(f"{window_extent[0]}x{window_extent[1]}")
        self.tk_root = tk_root

        tk_default_font = tkintFont.nametofont("TkDefaultFont")
        font_title = tk_default_font.copy()
        font_title.configure(size=12)

        root_frame = tkint.Frame(tk_root)
        root_frame.pack(padx=5, pady=5)

        ctx_tab_navbar = tkintTtk.Notebook(root_frame)
        ctx_tab_navbar.pack(side="top", fill="x")

        # TAB 1
        ctx_tab1 = tk.Frame(ctx_tab_navbar)
        ctx_tab_navbar.add(ctx_tab1, text="  Text analizer  ")
        root_frame = ctx_tab1

        tkint.Label(root_frame, text="Some title", justify="left", font=font_title).pack(padx=5, pady=5, anchor="w")
        tkint.Button(root_frame, text="Some button").pack(side="bottom")

        frame_contents = tkint.Frame(root_frame)
        frame_contents.pack()

        # Text input
        frame_text = tkint.Frame(frame_contents)
        frame_text.grid(column=0, row=0)
        ctx_text_area = tkint.Text(frame_text, width=64, height=16, wrap="word", autoseparators=True, undo=True, maxundo=-1)
        ctx_text_area.configure(tabs=tkintFont.Font(font=ctx_text_area["font"]).measure(" " * 4))  # tabsize 4
        ctx_text_area.pack(side="left")
        ctx_text_area.insert("1.0", _sample_text)
        ctx_text_area.bind("<KeyRelease>", self.on_TextAreaKeyRelease )
        self.ctx_text_area = ctx_text_area
        self._last_textarea_contents = ""
        self._last_textarea_model_reazults = None
        self._textarea_ttps : list[tktooltip.ToolTip] = list()

        # Scroll bar for text input
        ctx_text_slider = tkint.Scrollbar(frame_text, orient="vertical", command=ctx_text_area.yview)
        ctx_text_slider.pack(side="right", fill="y")
        ctx_text_area.configure(yscrollcommand=ctx_text_slider.set)

        # Model controll parameters
        frame_controlls = tkint.Frame(frame_contents, background="cyan")
        frame_controlls.grid(column=1, row=0, sticky="nsew")

        # Model treshold
        frame_parambox_1 = tkint.Frame(frame_controlls, padx=5, pady=5, background='', highlightbackground="black", highlightthickness=1)
        frame_parambox_1.grid(column=0, row=0, padx=5, pady=5, sticky='nsew')
        tkint.Label(frame_parambox_1, text="Model \ntreshold").grid(row=0, column=0)
        ctx_model_treshold = tkint.Scale(frame_parambox_1, from_=0.0, to=1.0, digits=3, resolution=0.01, orient="horizontal", length=150)
        ctx_model_treshold.grid(row=0, column=1)
        ctx_model_treshold.set(0.69)
        # ctx_model_treshold.configure(command= lambda v: self.on_RedoTextAreaHighlighting() ) # This is way to laggy
        ctx_model_treshold.bind("<ButtonRelease>", lambda e: self.on_RedoTextAreaHighlighting() )
        self.ctx_model_treshold = ctx_model_treshold

        # Model type
        frame_parambox_2 = tkint.Frame(frame_controlls, padx=5, pady=5, background='', highlightbackground="black", highlightthickness=1)
        frame_parambox_2.grid(column=0, row=1, padx=5, pady=5, sticky='nsew')
        tkint.Label(frame_parambox_2, text="Model type").grid(row=0, column=0)
        ctx_model_type = tkintTtk.Combobox(frame_parambox_2, state="readonly", values=["LR - Logical Regression", "RF - Random Forest", "math.Random()", "Chat-GPT", "Some Indian"])
        ctx_model_type.current(0)
        ctx_model_type.grid(row=0, column=1)
        ctx_model_type.bind("<<ComboboxSelected>>", lambda e: self.on_RedoTextAreaHighlighting() )
        self.ctx_model_type = ctx_model_type

        # Highlight full model output checkbox
        frame_parambox_3 = tkint.Frame(frame_controlls, padx=5, pady=5, background='', highlightbackground="black", highlightthickness=1)
        frame_parambox_3.grid(column=0, row=2, padx=5, pady=5, sticky='nsew')
        var_do_highlight = tkint.IntVar()
        ctx_do_highlight = tkint.Checkbutton(frame_parambox_3, onvalue=1, offvalue=0, text="Continuios highlight")
        ctx_do_highlight.grid(row=0, column=0)
        ctx_do_highlight.configure(variable= var_do_highlight, command= self.on_RedoTextAreaHighlighting )
        self.ctx_do_highlight = ctx_do_highlight
        self.var_do_highlight = var_do_highlight

        # Reset params to default state
        ctx_button_reset_param = tkint.Button(frame_controlls, text="Reset parameters")
        ctx_button_reset_param.configure(command= self.on_ResetModelParameter )
        ctx_button_reset_param.grid(padx=5, pady=5, sticky='nsew')

        # Debug show tokenization output
        ctx_button_dbg_tokens = tkint.Button(frame_controlls, text="Show tokenization")
        ctx_button_dbg_tokens.configure( command= self.on_HighlightTokenization )
        ctx_button_dbg_tokens.grid(padx=5, pady=5, sticky='nsew')

        # Manual eval button or something
        ctx_button_do_funny = tkint.Button(frame_controlls, text="Do the funny")
        ctx_button_do_funny.configure(command= lambda: log.debug("Manual update") or self.on_RedoTextAreaHighlighting() )
        ctx_button_do_funny.grid(padx=5, pady=5, sticky='nsew')

        # import tktooltip
        # ctx_ttip = tktooltip.ToolTip(ctx_button_do_funny, lambda: f"Mordoria {random.randint(0, 100)}")

        # # TAB 2
        # ctx_tab2 = tk.Frame(ctx_tab_navbar)
        # ctx_tab_navbar.add(ctx_tab2, text="  TAB 2  ")
        # root_frame = ctx_tab2
        #
        # tkint.Label(
        #     root_frame, text="Literary a second tab, this shit is insane", justify="left", font=font_title
        # ).pack(padx=5, pady=5, anchor="w")

        self._dbg_highlight = False
        return

    def loop_blocking(self):
        self.tk_root.mainloop()

    # Helper UI functions

    @staticmethod
    def tokenize_text(text: str) -> Generator[tuple[int, int, str], None, None]:
        for re_token in re.finditer(r"\b\w[\w'-]*\b", text):
            start = re_token.start()
            end = re_token.end()
            token_str = re_token.group()
            yield (start, end, token_str)
        return None

    def highlight_textarea(self, text_contents: str | None = None):
        ctx_text_area = self.ctx_text_area

        def _weighting_functon(word: str) -> float:
            return min(1.0, (len(word) / 10.0))

        def _prob_to_color(prob: float) -> str:
            color1 = (1.0, 0.0, 0.0)
            color2 = (0.0, 1.0, 0.0)
            mixed = (
                int( 0xFF * pow(color1[0] * (1 - prob) + color2[0] * prob, 1.0 / 2.2) ),
                int( 0xFF * pow(color1[1] * (1 - prob) + color2[1] * prob, 1.0 / 2.2) ),
                0x00
            )
            return "#{:02x}{:02x}00".format(mixed[0], mixed[1])

        if text_contents is None:
            new_contents = ctx_text_area.get("1.0", tk.END)
        else:
            new_contents = text_contents

        # Strip all current formating
        for tag in ctx_text_area.tag_names():
            ctx_text_area.tag_delete(tag)
        for ttp in self._textarea_ttps:
            ttp.destroy()
        self._textarea_ttps.clear()

        # Perform text tagging if input has changed
        model_treshold = self.ctx_model_treshold.get()
        if self._last_textarea_model_reazults is not None and self._last_textarea_contents == new_contents:
            log.debug("Reusing results")
            tokenized, probabilities = self._last_textarea_model_reazults
        else:
            self._last_textarea_contents = new_contents
            tokenized = list(Application.tokenize_text(new_contents))
            probabilities = [ _weighting_functon(token) for _, _, token in tokenized ]
            self._last_textarea_model_reazults = tokenized, probabilities

        # Apply tags to textarea
        continuos_highlight = (self.var_do_highlight.get() == 1)
        for i, (idx_start, idx_end, token) in enumerate(tokenized):
            start_idx = f"1.0 + {idx_start} chars"
            end_idx = f"1.0 + {idx_end} chars"

            # Word formating
            tag_name = f"word_{i}"
            word_prob = probabilities[i]
            ctx_text_area.tag_add(tag_name, start_idx, end_idx)
            if continuos_highlight:
                ctx_text_area.tag_config(tag_name, background=_prob_to_color(word_prob))
            elif model_treshold < word_prob:
                ctx_text_area.tag_config(tag_name, underline=True, underlinefg="red")

            # Word tooltip
            _proxy: tkint.Widget = _TextTagBindingProxy(ctx_text_area, tag_name) # noqa
            ttp = tktooltip.ToolTip(_proxy, f"{token}: {word_prob:.2f}" )
            self._textarea_ttps.append(ttp)

        return

    # UI callbacks

    def on_ResetModelParameter(self):
        log.error("on_ResetModelParameter")
        return

    def on_RedoTextAreaHighlighting(self):
        log.debug("on_RedoTextAreaHighlighting")
        self.highlight_textarea()

    def on_TextAreaKeyRelease(self, event: tkint.Event):
        # Only update if last keybind didnt make any textual changes
        if event.char != "":
            return

        if self._dbg_highlight:
            return

        new_contents = self.ctx_text_area.get("1.0", tk.END)
        if self._last_textarea_contents == new_contents:
            return

        self.highlight_textarea(new_contents)

    def on_HighlightTokenization(self):
        ctx_text_area = self.ctx_text_area

        # Strip all current formating
        for tag in ctx_text_area.tag_names():
            ctx_text_area.tag_delete(tag)

        # Turn off highlighting if already on
        if self._dbg_highlight:
            self._dbg_highlight = False
            return

        contetns = ctx_text_area.get("1.0", tkint.END)
        for i, (idx_start, idx_end, token) in enumerate(Application.tokenize_text(contetns)):
            # print(idx_start, idx_end, token)
            start_idx = f"1.0 + {idx_start} chars"
            end_idx = f"1.0 + {idx_end} chars"

            # color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            color = "#FF7777"
            tag_name = f"word_{i}"
            ctx_text_area.tag_add(tag_name, start_idx, end_idx)
            ctx_text_area.tag_config(tag_name, background=color)
            # ctx_text_area.tag_bind(tag_name, start_idx, end_idx)

            # ctx_text_area.tag_bind(tag_name, "<Enter>", make_enter())
            # ctx_text_area.tag_bind(tag_name, "<Leave>", leave)

            _proxy: tkint.Widget = _TextTagBindingProxy(ctx_text_area, tag_name) # noqa
            ttp = tktooltip.ToolTip(_proxy, f"{(idx_start, idx_end, token)}" )

        self._dbg_highlight = True
        return

    ...


def main():
    log.info("Start of main")

    app = Application()
    app.loop_blocking()

    return


if __name__ == '__main__':
    main()
