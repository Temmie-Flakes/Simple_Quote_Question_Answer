A rushed implementation and UI of question-answering models. currently only tested on deepset/roberta-base-squad2<br />

quick install (windows):<br />
----
1. download this repo as a zip or using Git.
2. download and install [python 3.10.6](https://www.python.org/downloads/release/python-3106/) (recommended) and add to PATH (untested on later versions including 3.11.3)
4. run Install.bat and wait for it to finish and say `Press any key to continue . . .`
5. run RunBaseModel.bat

how to use:
----
1. put the question in the question box and the context in the context box.
2. hit "Do Stuff"

you can add the flag `--repo-id [hugging face repo name]` line in the batch file to use a different model other then deepset/roberta-base-squad2.

Notes:<br />
----
- If you don't have a cuda enabled GPU, edit RunBaseModel.bat and change:<br />
`venv\Scripts\python.exe Open_WebUI.py --cache_dir "modelsCache"` <br />
to:<br />
`venv\Scripts\python.exe Open_WebUI.py --cache_dir "modelsCache" --device "cpu"`<br />


Todo:
----
- [ ] Fix everything