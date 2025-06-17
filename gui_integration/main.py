
import os
import sys
import random
import functools
import threading
import concurrent.futures
import dataclasses as dc
from typing import Generator, Callable, Any

import re

# python tkinter docks : "https://anzeljg.github.io/rin2/book2/2405/docs/tkinter/"
# sudo apt-get install python3-tk  Wtf linux? Idk how to add this dependency to the project config

import tkinter as tkint
import tkinter.ttk as tkintTtk
import tkinter.font as tkintFont
import tktooltip
tk = tkint

import joblib
import numpy as np
import pandas as pd
from classifier.train import LoanwordClassifier

import gui_integration.utils as utils
import gui_integration.samples as text_samples

log = utils.get_logger(__name__)


@dc.dataclass(slots=True, init=False)
class ModelParams:
    model_name: str
    classifier_obj: LoanwordClassifier
    default_treshold: float
    custom_trehsold: float

    def __init__(self, model_name: str, model_file_path: str):
        assert os.path.isfile(model_file_path), "No such file: {}".format(model_file_path)
        self.model_name: str = model_name
        self.classifier_obj = joblib.load(model_file_path)
        self.custom_trehsold  = self.classifier_obj.threshold
        self.default_treshold = self.classifier_obj.threshold

    def PredictProbabilities(self, words: list[str]) -> list[float]:
        # Dataframe because we are too cool to use std::list
        df_words = pd.DataFrame({"word": words})

        # Vectorize the input words
        X, _ = self.classifier_obj.vectorize_words(df_words)

        # Predict using the model
        probabilities: np.ndarray = self.classifier_obj.predict_proba(X)
        return probabilities # noqa "This is fine"


class DummyModlelParams(ModelParams):
    def __init__(self, model_name: str, eval_func: Callable[[str], float]):
        log.warning(f"DummyModlelParams '{model_name}' created.")
        try: super().__init__(model_name, "")
        except AssertionError: pass
        self.model_name = model_name
        self.custom_trehsold  = 0.69
        self.default_treshold = 0.69
        self._eval_func = eval_func

    def PredictProbabilities(self, words: list[str]) -> list[float]:
        return [ self._eval_func(word) for word in words ]


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
        self._models: dict[str, ModelParams] = dict()
        self._load_models()

        window_extent = (800, 450)
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

        # Examples
        tkint.Label(root_frame, text="Some title", justify="left", font=font_title).pack(padx=5, pady=5, anchor="w")

        # Example sample text buttons
        frame_parambox_42 = tkint.Frame(root_frame, padx=5, pady=5, background='', highlightbackground="black", highlightthickness=1)
        frame_parambox_42.pack(side="bottom", fill="x")
        tkint.Label(frame_parambox_42, text="Examples:") .pack(side="left")
        for sample_name, sample_text in text_samples.sample_text_samples:
            ctx_sample_btn = tkint.Button(frame_parambox_42, text=sample_name)
            ctx_sample_btn.pack(side="left", padx=5, pady=1)
            ctx_sample_btn.config(command= functools.partial(self.on_SetSampleText, sample_name, sample_text) )

        # Main UI contents
        frame_contents = tkint.Frame(root_frame)
        frame_contents.pack()

        # Text input
        frame_text = tkint.Frame(frame_contents)
        frame_text.grid(column=0, row=0)
        ctx_text_area = tkint.Text(frame_text, width=64, height=16, wrap="word", autoseparators=True, undo=True, maxundo=-1)
        ctx_text_area.configure(tabs=tkintFont.Font(font=ctx_text_area["font"]).measure(" " * 4))  # tabsize 4
        ctx_text_area.pack(side="left")
        ctx_text_area.bind("<KeyRelease>", self.on_TextAreaKeyRelease )
        if text_samples.default_sample is not None:
            ctx_text_area.insert("1.0", text_samples.get_default_sample_text() )

        self.ctx_text_area = ctx_text_area
        self._last_textarea_contents = ""
        self._last_textarea_tokenized = None
        self._last_textarea_probabilities = None
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
        ctx_model_treshold.bind("<ButtonRelease>", lambda e: self.on_ModelTresholdChange() )
        self.ctx_model_treshold = ctx_model_treshold

        # Model type
        frame_parambox_2 = tkint.Frame(frame_controlls, padx=5, pady=5, background='', highlightbackground="black", highlightthickness=1)
        frame_parambox_2.grid(column=0, row=1, padx=5, pady=5, sticky='nsew')
        tkint.Label(frame_parambox_2, text="Model type").grid(row=0, column=0)
        ctx_model_type = tkintTtk.Combobox(
            frame_parambox_2, state="readonly", values= [ m_name for m_name in self._models.keys() ]
        )
        ctx_model_type.current(0)
        ctx_model_type.grid(row=0, column=1)
        ctx_model_type.bind("<<ComboboxSelected>>", lambda e: self.on_ModelTypeChange() )
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
        ctx_button_reset_param.configure(command= self.on_ResetModelParameters)
        ctx_button_reset_param.grid(padx=5, pady=5, sticky='nsew')

        # Debug show tokenization output
        ctx_button_dbg_tokens = tkint.Button(frame_controlls, text="Show tokenization")
        ctx_button_dbg_tokens.configure( command= self.on_HighlightTokenization )
        ctx_button_dbg_tokens.grid(padx=5, pady=5, sticky='nsew')
        self._dbg_highlight = False

        # Manual eval button or something
        ctx_button_do_funny = tkint.Button(frame_controlls, text="Manual highlight")
        ctx_button_do_funny.configure(command= lambda: log.debug("Manual update") or self.on_RedoTextAreaHighlighting() )
        ctx_button_do_funny.grid(padx=5, pady=5, sticky='nsew')

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._highlight_lock = threading.Lock()

        # # TAB 2
        # ctx_tab2 = tk.Frame(ctx_tab_navbar)
        # ctx_tab_navbar.add(ctx_tab2, text="  TAB 2  ")
        # root_frame = ctx_tab2
        #
        # tkint.Label(
        #     root_frame, text="Literary a second tab, this shit is insane", justify="left", font=font_title
        # ).pack(padx=5, pady=5, anchor="w")

        self.on_ResetModelParameters()
        return

    def loop_blocking(self):
        self.tk_root.mainloop()

    def _load_models(self):
        this_path = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(this_path, 'packaged_models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # todo: Download models if it doesnt exist
        model_args = [
            (ModelParams, "RF - Random Forest",      os.path.join(models_dir, "rf_v0_2_1.pkl")),
            (ModelParams, "LR - Logical Regression", os.path.join(models_dir, "lr_v0_2_1.pkl")),
        ]

        self._models = {
            model_name:
                log.info(f"Loading model: {model_name}, from {os.path.basename(model_path)}")
                or _LoaderClass(model_name, model_path)
            for _LoaderClass, model_name, model_path in model_args
        }

        self._models[_dm1.model_name] = (
            _dm1 := DummyModlelParams("Dummy model", lambda _w: min(1.0, (len(_w) / 10.0)) )
        )
        return

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
        current_model = self.ctx_model_type.get()

        model_params = self._models.get(current_model, None)
        if model_params is None:
            log.warning(f"Model {current_model} not found, what?")
            return

        if text_contents is None:
            new_contents = ctx_text_area.get("1.0", tk.END)
        else:
            new_contents = text_contents

        with self._highlight_lock:
            # Do we need to retokenize?
            do_contents_match = (self._last_textarea_contents == new_contents)
            self._last_textarea_contents = new_contents

            # Perform text tagging if input has changed
            if self._last_textarea_tokenized is not None and do_contents_match:
                tokenized = self._last_textarea_tokenized
            else:
                tokenized = list(Application.tokenize_text(new_contents))
                self._last_textarea_tokenized = tokenized

            # Do we need to recompute probabilities?
            if self._last_textarea_probabilities is not None and do_contents_match:
                probabilities = self._last_textarea_probabilities
                self._apply_highlighting(tokenized, probabilities)
                return

            only_words = [token for _, _, token in tokenized]
            if not only_words:
                self._last_textarea_probabilities = []
                self._apply_highlighting(tokenized, [])
                return

            future = self._executor.submit(model_params.PredictProbabilities, only_words)
            future.add_done_callback(lambda fut: self.tk_root.after(
                0, self._on_prediction_ready, tokenized, fut
            ))

    def _on_prediction_ready(self, tokenized: list[tuple[int, int, str]], fut: concurrent.futures.Future):
        try:
            probabilities = fut.result()
        except Exception as e:
            log.exception("Error during prediction: %s", e)
            return

        with self._highlight_lock:
            self._last_textarea_probabilities = probabilities
            self._apply_highlighting(tokenized, probabilities)

    def _apply_highlighting(self, tokenized: list[tuple[int, int, str]], probabilities: list[float]):
        ctx_text_area = self.ctx_text_area

        # Strip all current formating
        for tag in ctx_text_area.tag_names():
            ctx_text_area.tag_delete(tag)
        for ttp in self._textarea_ttps:
            ttp.destroy()
        self._textarea_ttps.clear()

        # Apply tags to textarea
        model_treshold = self.ctx_model_treshold.get()
        continuos_highlight = (self.var_do_highlight.get() == 1)
        for i, (idx_start, idx_end, token) in enumerate(tokenized):
            start_idx = f"1.0 + {idx_start} chars"
            end_idx = f"1.0 + {idx_end} chars"

            # Word formating
            tag_name = f"word_{i}"
            word_prob = probabilities[i]
            ctx_text_area.tag_add(tag_name, start_idx, end_idx)
            if continuos_highlight:
                ctx_text_area.tag_config(tag_name, background=utils.prob_to_color(word_prob))
            elif model_treshold < word_prob:
                ctx_text_area.tag_config(tag_name, underline=True, underlinefg="red")

            # Word tooltip
            _proxy: tkint.Widget = _TextTagBindingProxy(ctx_text_area, tag_name) # noqa, Trust me bro, this is the righ type
            ttp = tktooltip.ToolTip(_proxy, f"{token}: {word_prob:.2f}" )
            self._textarea_ttps.append(ttp)

        return

    # UI callbacks

    def on_SetSampleText(self, sample_name: str, sample_text: str):
        log.debug(f"on_SetSampleText {sample_name}")
        self.ctx_text_area.delete("1.0", "end")
        self.ctx_text_area.insert("1.0", sample_text)
        self.on_RedoTextAreaHighlighting()

    def on_ResetModelParameters(self):
        log.error("on_ResetModelParameters")
        current_model = self.ctx_model_type.get()

        model_params = self._models.get(current_model, None)
        if model_params is None:
            log.warning(f"Model {current_model} not found, what?")
            return

        model_params.custom_trehsold = model_params.default_treshold
        self.ctx_model_treshold.set(model_params.custom_trehsold)
        self.var_do_highlight.set(0)
        self.on_RedoTextAreaHighlighting()

    def on_ModelTypeChange(self):
        log.debug("on_ModelTypeChange")
        current_model = self.ctx_model_type.get()

        model_params = self._models.get(current_model, None)
        if model_params is None:
            log.warning(f"Model {current_model} not found, what?")
            return

        self.ctx_model_treshold.set(model_params.custom_trehsold)
        self._last_textarea_probabilities = None # Invalidate cached probs.
        self.on_RedoTextAreaHighlighting()

    def on_ModelTresholdChange(self):
        log.debug("on_ModelTresholdChange")
        new_treshold = self.ctx_model_treshold.get()
        current_model = self.ctx_model_type.get()

        model_params = self._models.get(current_model, None)
        if model_params is None:
            log.warning(f"Model {current_model} not found, what?")
            return

        model_params.custom_trehsold = new_treshold
        self.on_RedoTextAreaHighlighting()

    def on_RedoTextAreaHighlighting(self):
        log.debug("on_RedoTextAreaHighlighting")
        self.highlight_textarea()

    def on_TextAreaKeyRelease(self, event: tkint.Event):
        # Only update if last keybind didnt make any textual changes
        if event.char != "":
            return
        if event.keysym == "BackSpace":
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
            color = "#FF8888"
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



def main():
    log.info("Start of main")

    app = Application()
    app.loop_blocking()

    return


if __name__ == '__main__':
    main()
