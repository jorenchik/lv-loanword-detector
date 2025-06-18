

sample_text_samples: list[tuple[str, str]] = list()
def reg_new(name: str, text: str) -> None:
    sample_text_samples.append((name, text.lstrip('\n') ))

# default_sample = "Valus Ta'aurc"

# Example LV text 1
reg_new("Sample LV 1", """
Rītā Antra devās uz universitāti, lai sagatavotu prezentāciju. 
Viņa izmantoja kompiuteri, internetu un printeri, lai atrastu 
nepieciešamo informāciju. Pēc tam viņa apmeklēja kafejnīcu, kur 
pasūtīja kafiju un kruasānu. Darba diena bija produktīva, jo 
visi projekta uzdevumi tika izpildīti efektīvi un savlaicīgi.
""")

# Example LV text 2
reg_new("Sample LV 2", """
No rīta Inese devās uz laboratoriju, lai sagatavotu eksperimenta 
protokolu. Viņa izmantoja mikroskopu, spektrametru un analītisko 
programmatūru, lai apstrādātu un analizētu iegūtos datus. Visi 
parametri tika ierakstīti datubāzē un saglabāti serverī. Pēc tam 
viņa sagatavoja prezentāciju starptautiskajai konferencei, kurā 
apsprieda inovācijas biotehnoloģiju sektorā.
""")

# ENG text - Quote from D2
reg_new("Valus Ta'aurc", """
Whether we wanted it or not, we've stepped into a war with the
Cabal on Mars. So let's get to taking out their command, one by
one. Valus Ta'aurc. From what I can gather he commands the Siege
Dancers from an Imperial Land Tank outside of Rubicon. He's well 
protected, but with the right team, we can punch through those
defenses, take this beast out, and break their grip on Freehold.
""")


del reg_new
default_sample = None if "default_sample" not in locals() else locals()["default_sample"]
if default_sample is not None:
    assert any( s[0] == default_sample for s in sample_text_samples ), "Default sample is not real >>" + default_sample

def get_default_sample_text() -> str:
    return next(s for s in sample_text_samples if s[0] == default_sample)[1]
