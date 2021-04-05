from teletext.cli.teletext import teletext
from teletext.cli.training import training

teletext.add_command(training)

if __name__ == '__main__':
    teletext()
