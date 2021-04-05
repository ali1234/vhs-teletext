import unittest

from click.testing import CliRunner

import teletext.cli.teletext
import teletext.cli.training


class TestCommandTeletext(unittest.TestCase):
    cmd = teletext.cli.teletext.teletext

    def setUp(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(self.cmd, ['--help'])
        self.assertEqual(result.exit_code, 0)


class TestCmdFilter(TestCommandTeletext):
    cmd = teletext.cli.teletext.filter


class TestCmdDiff(TestCommandTeletext):
    cmd = teletext.cli.teletext.diff


class TestCmdFinders(TestCommandTeletext):
    cmd = teletext.cli.teletext.finders


class TestCmdSquash(TestCommandTeletext):
    cmd = teletext.cli.teletext.squash


class TestCmdSpellcheck(TestCommandTeletext):
    cmd = teletext.cli.teletext.spellcheck


class TestCmdService(TestCommandTeletext):
    cmd = teletext.cli.teletext.service


class TestCmdInteractive(TestCommandTeletext):
    cmd = teletext.cli.teletext.interactive


class TestCmdUrls(TestCommandTeletext):
    cmd = teletext.cli.teletext.urls


class TestCmdHtml(TestCommandTeletext):
    cmd = teletext.cli.teletext.html


class TestCmdRecord(TestCommandTeletext):
    cmd = teletext.cli.teletext.record


class TestCmdVBIView(TestCommandTeletext):
    cmd = teletext.cli.teletext.vbiview


class TestCmdDeconvolve(TestCommandTeletext):
    cmd = teletext.cli.teletext.deconvolve


class TestCmdTraining(TestCommandTeletext):
    cmd = teletext.cli.training.training


class TestCmdGenerate(TestCommandTeletext):
    cmd = teletext.cli.training.generate


class TestCmdTrainingSquash(TestCommandTeletext):
    cmd = teletext.cli.training.training_squash


class TestCmdShowBin(TestCommandTeletext):
    cmd = teletext.cli.training.showbin


class TestCmdBuild(TestCommandTeletext):
    cmd = teletext.cli.training.build
