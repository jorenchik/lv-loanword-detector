
import os
import sys

# sudo apt-get install python3-tk  Wtf linux? Idk how to add this dependency to the project config
import tkinter as tkint
import tkinter.ttk as tkintTtk
import tkinter.font as tkintFont
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


def main():
    log.info("Start of main")

    window_extent = (800, 600)
    tk_root = tkint.Tk()
    tk_root.title("mordoria")
    tk_root.geometry( f"{window_extent[0]}x{window_extent[1]}" )

    tk_default_font = tkintFont.nametofont("TkDefaultFont")
    font_title = tk_default_font.copy()
    font_title.configure(size=12)

    root_frame = tkint.Frame(tk_root)
    root_frame.pack(padx=5, pady=5)

    ctx_tab_navbar = tkintTtk.Notebook(root_frame)
    ctx_tab_navbar.pack(side="top", fill="x")


    # TAB 1
    ctx_tab1 = tk.Frame(ctx_tab_navbar)
    ctx_tab_navbar.add(ctx_tab1, text="  TAB 1  ")
    root_frame = ctx_tab1

    tkint.Label(root_frame, text="Some title", justify="left", font=font_title).pack(padx=5, pady=5, anchor="w")
    tkint.Button(root_frame, text="Some button").pack(side="bottom")

    frame_contents = tkint.Frame(root_frame)
    frame_contents.pack()

    frame_text = tkint.Frame(frame_contents)
    frame_text.grid(column=0, row=0)
    ctx_text_area = tkint.Text(
        frame_text, width=64, height=16, wrap="word", autoseparators=True, undo=True, maxundo=-1
    )
    ctx_text_area.configure( tabs= tkintFont.Font( font=ctx_text_area["font"] ).measure(" " * 4) ) # tabsize 4
    ctx_text_area.pack(side="left")
    ctx_text_area.insert("1.0", _sample_text)

    ctx_text_slider = tkint.Scrollbar(frame_text, orient="vertical", command=ctx_text_area.yview)
    ctx_text_slider.pack(side="right", fill="y")
    ctx_text_area.configure(yscrollcommand=ctx_text_slider.set)

    frame_controlls = tkint.Frame(frame_contents, background="cyan")
    frame_controlls.grid(column=1, row=0, sticky="nsew")

    frame_parambox_1 = tkint.Frame(
        frame_controlls, padx=5, pady=5, background='', highlightbackground="black", highlightthickness=1)
    frame_parambox_1.grid(column=0, row=0, padx=5, pady=5, sticky='nsew')
    tkint.Label(frame_parambox_1, text="Model \ntreshold").grid(row=0, column=0)
    ctx_model_scale = tkint.Scale(frame_parambox_1, from_=0.0, to=1.0, digits=3, resolution=0.01, orient="horizontal", length=150)
    ctx_model_scale.grid(row=0, column=1)
    ctx_model_scale.set(0.69)

    frame_parambox_2 = tkint.Frame(
        frame_controlls, padx=5, pady=5, background='', highlightbackground="black", highlightthickness=1)
    frame_parambox_2.grid(column=0, row=1, padx=5, pady=5, sticky='nsew')

    tkint.Label(frame_parambox_2, text="Model type").grid(row=0, column=0)
    ctx_model_type = tkintTtk.Combobox(
        frame_parambox_2, state="readonly", values=["LR - Logical Regression", "RF - Random Forest", "math.Random()", "Chat-GPT", "Some Indian"] )
    ctx_model_type.current(0)
    ctx_model_type.grid(row=0, column=1)


    # TAB 2
    ctx_tab2 = tk.Frame(ctx_tab_navbar)
    ctx_tab_navbar.add(ctx_tab2, text="  TAB 2  ")
    root_frame = ctx_tab2

    tkint.Label(
        root_frame, text="Literary a second tab, this shit is insane", justify="left", font=font_title).pack(padx=5, pady=5, anchor="w")



    tk_root.mainloop()

    ...

if __name__ == '__main__':
    main()
