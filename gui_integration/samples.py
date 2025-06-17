

sample_text_samples: list[tuple[str, str]] = list()
def reg_new(name: str, text: str) -> None:
    sample_text_samples.append((name, text.lstrip('\n') ))

default_sample = "Valus Ta'aurc"

# Quote from D2
reg_new("Valus Ta'aurc", """
Whether we wanted it or not, we've stepped into a war with the
Cabal on Mars. So let's get to taking out their command, one by
one. Valus Ta'aurc. From what I can gather he commands the Siege
Dancers from an Imperial Land Tank outside of Rubicon. He's well 
protected, but with the right team, we can punch through those
defenses, take this beast out, and break their grip on Freehold.
""")

reg_new("Shitpost 1", """
Shitpost 1
""")
reg_new("Shitpost 2", """
Shitpost 2
""")
reg_new("Shitpost 3", """
Shitpost 3
""")

del reg_new
if default_sample is not None:
    assert any( s[0] == default_sample for s in sample_text_samples ), "Default sample is not real >>" + default_sample

def get_default_sample_text() -> str:
    return next(s for s in sample_text_samples if s[0] == default_sample)[1]
