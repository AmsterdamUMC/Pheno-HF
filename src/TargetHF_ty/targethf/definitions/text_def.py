# Some sub-patterns useful for constructing complex patterns 
no_denial   = r"(?<!\bgeen )(?<!\bniet )(?<!\bzonder )"
no_doubt    = r"(?<!\bmogelijk )(?<!\bverdenking )(?<!\bcave )(?<!\bdd )(?<!\bd\.d\. )"
no_fear     = r"(?<!\bangst voor )(?<!\bschrik voor )"
no_question = r"(?! *\?)"
boundary    = r"(\b)"

# Patterns to search for, organized per condition

dyspnea =            ["OR", no_denial + boundary + r"d[iy]spn",
                            no_denial + boundary + r"benauw(d|ig|end)",
                            no_denial + boundary + r"beklem(d|mend)",
                            no_denial + boundary + r"drukkend",
                            no_denial + boundary + r"kortademig",
                            no_denial + boundary + r"kort +v(\.|an)? +adem", 
                            no_denial + boundary + r"adem *te *kort"]

edema =              ["OR",  no_denial + boundary + r"enkeloede[em]",
                            ["AND",       ["OR", no_denial + boundary + r"vocht(ophoping|\b)",
                                                 no_denial + boundary + r"(op)?gezwollen",
                                                 no_denial + boundary + r"oede[em]",
                                                 no_denial + boundary + r"dikke"],
                                          ["OR", boundary + r"benen"  + boundary,
                                                 boundary + r"enkels" + boundary]
                            ]
                     ]

heart_failure =      ["OR", no_denial + no_doubt + no_fear + r"hart *falen"     + no_question,
                            no_denial + no_doubt + no_fear + r"cardio *m[yi]opath?ie"      + no_question,
                            no_denial + no_doubt + no_fear + r"dec(\.?|omp\w*\.?) *cordis" + no_question]



afib =      ["OR", no_denial + no_doubt + no_fear + r"atrium\s*f"  + no_question,
                            no_denial + no_doubt + no_fear + r"fibr(?!o)" + no_question,
                            no_denial + no_doubt + no_fear + boundary + r"boez" + no_question,
                            no_denial + no_doubt + no_fear + boundary + r"[ab]fib" + boundary + no_question,
                            no_denial + no_doubt + no_fear + boundary + r"p?[ab]f" + boundary + no_question]

valvhd = ["OR", no_denial + no_doubt + no_fear + boundary + r"[am]?klep" + no_question,
                            no_denial + no_doubt + no_fear + boundary + r"valv" + no_question,
                            no_denial + no_doubt + no_fear + boundary + r"mitr" + no_question,
                            no_denial + no_doubt + no_fear + boundary + r"aorta\s*k.*" + no_question,
                            no_denial + no_doubt + no_fear + boundary +  r"insuf.*" + no_question, 
                            # blacklist icpcs
                            # ["F99.06",  "K99.04",  "T99.12",  "U99.01",  "U99",  "W84.06"]
                            no_denial + no_doubt + no_fear + r"steno.*" + no_question
                            # ["D81.03", "D84.05", "F80", "H73"]
                            ]

