
import os
import sys

# sudo apt-get install python3-tk  Wtf linux? Idk how to add this dependency to the project config
import tkinter as tkint
import tkinter.font as tkintFont

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

    # frame = tkint.Frame(tk_root, width=300, height=300)
    root_frame = tkint.Frame(tk_root)
    root_frame.grid(padx=10, pady=10)
    tkint.Label(root_frame, text="mordoria").grid(row=0, column=0)
    tkint.Button(root_frame, text="Some button").grid(row=1, column=0)

    frame_text = tkint.Frame(root_frame)
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


    frame_controlls = tkint.Frame(root_frame, background="cyan")
    frame_controlls.grid(column=1, row=0, sticky="nsew")
    tkint.Label(frame_controlls, text="mordoria").grid(row=0, column=0)


    frame_parambox_1 = tkint.Frame(
        frame_controlls, padx=5, pady=5, background='', highlightbackground="black", highlightthickness=1)
    frame_parambox_1.grid(column=0, row=0, padx=5, pady=5)
    tkint.Label(frame_parambox_1, text="Model \ntreshold ").grid(row=0, column=0)
    ctx_model_scale = tkint.Scale(frame_parambox_1, from_=0.0, to=1.0, digits=3, resolution=0.01, orient="horizontal", length=150)
    ctx_model_scale.grid(row=0, column=1)
    ctx_model_scale.set(0.69)

    # pw_label = Label(root, text="Password").grid(row=0, column=0, pady=4, padx=4)
    # length_label = Label(root, text="Length").grid(row=2, column=0, pady=4, padx=4)
    # pw_input = Entry(root, width=50)
    # length_input = Scale(root, from_=8, to=50, orient=HORIZONTAL, length=300)
    # length_input.set(30)
    # pw_input.grid(row=0, column=1, pady=4, padx=4)
    # length_input.grid(row=2, column=1, pady=4, padx=4)
    # Button(root, text='Quit', command=root.quit).grid(row=3, column=2, sticky=W, pady=4)
    # Button(root, text='Generate', command=rand_pw).grid(row=2, column=2, sticky=W, pady=4)
    # Button(root, text='Copy', command=copy).grid(row=0, column=2, sticky=W, pady=4)

    tk_root.mainloop()

    ...

if __name__ == '__main__':
    main()
