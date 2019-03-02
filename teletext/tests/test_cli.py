import unittest
import sys

from click.testing import CliRunner

import teletext.cli


class TestCommandTeletext(unittest.TestCase):
    cmd = teletext.cli.teletext

    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(self.cmd, ['--help'])
        self.assertEqual(result.exit_code, 0)


class TestCmdFilter(TestCommandTeletext):
    cmd = teletext.cli.filter


class TestCmdDiff(TestCommandTeletext):
    cmd = teletext.cli.diff


class TestCmdFinders(TestCommandTeletext):
    cmd = teletext.cli.finders


class TestCmdSquash(TestCommandTeletext):
    cmd = teletext.cli.squash


class TestCmdSpellcheck(TestCommandTeletext):
    cmd = teletext.cli.spellcheck


class TestCmdService(TestCommandTeletext):
    cmd = teletext.cli.service


class TestCmdInteractive(TestCommandTeletext):
    cmd = teletext.cli.interactive


class TestCmdUrls(TestCommandTeletext):
    cmd = teletext.cli.urls


class TestCmdHtml(TestCommandTeletext):
    cmd = teletext.cli.html


class TestCmdRecord(TestCommandTeletext):
    cmd = teletext.cli.record


class TestCmdVBIView(TestCommandTeletext):
    cmd = teletext.cli.vbiview


class TestCmdDeconvolve(TestCommandTeletext):
    cmd = teletext.cli.deconvolve


class TestCmdTraining(TestCommandTeletext):
    cmd = teletext.cli.training


class TestCmdGenerate(TestCommandTeletext):
    cmd = teletext.cli.generate


class TestCmdTrainingSquash(TestCommandTeletext):
    cmd = teletext.cli.training_squash


class TestCmdShowBin(TestCommandTeletext):
    cmd = teletext.cli.showbin


class TestCmdBuild(TestCommandTeletext):
    cmd = teletext.cli.build
