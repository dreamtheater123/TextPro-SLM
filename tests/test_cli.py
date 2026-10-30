"""Installed entry points should be usable before downloading a model."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from textpro_slm import cli


def test_module_help_does_not_load_checkpoints():
    result = subprocess.run([sys.executable, "-m", "textpro_slm", "--help"],
                            capture_output=True, text=True, check=True)
    assert "--model" in result.stdout and "--audio" in result.stdout


@pytest.mark.parametrize("options,expected", [
    ([], {"instruction": "", "task": "semantic", "max_new_tokens": 512,
          "do_sample": True, "temperature": .7, "top_p": .9}),
    (["--instruction", "Question and choices", "--task", "emotion", "--temperature", ".6", "--top-p", ".8"],
     {"instruction": "Question and choices", "task": "emotion", "max_new_tokens": 512,
      "do_sample": True, "temperature": .6, "top_p": .8}),
    (["--greedy"], {"instruction": "", "task": "semantic", "max_new_tokens": 512,
                     "do_sample": False, "temperature": .7, "top_p": .9}),
])
def test_cli_passes_the_selected_input_mode_and_decoding_options(options, expected, capsys):
    # Exercise argparse -> public API without loading a checkpoint.
    respond = Mock(return_value={"response": "A"})
    with patch.object(sys, "argv", ["textpro-slm", "--audio", "speech.wav", *options]), \
         patch.object(cli, "TextProSLM", return_value=SimpleNamespace(respond=respond)):
        cli.main()
    respond.assert_called_once_with(audio="speech.wav", **expected)
    assert '"response": "A"' in capsys.readouterr().out
