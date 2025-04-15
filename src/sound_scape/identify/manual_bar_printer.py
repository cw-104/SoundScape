import time, shutil


def get_terminal_max_inline_characters():
    # Get the size of the terminal window
    terminal_size = shutil.get_terminal_size()

    # Get the width and height
    return terminal_size.columns


class manual_bar_printer:
    def __init__(
        self, total, max_bars=40, desc="", unit="file", none_char=" ", auto_size=True
    ):
        self.total = total
        # add space to end if not present
        self.desc = desc + (" " if desc != "" and not desc.endswith(" ") else "")

        self.unit = unit
        self.num_complete = 0
        self.start_time = time.time()
        self.none_char = none_char

        if auto_size:
            # self.max_bars = get_terminal_max_inline_characters() - len(self.desc) - len(self.unit) - 50
            self.max_bars = (
                get_terminal_max_inline_characters()
                - len(self.desc)
                - len("| xx/xx (xxx%)|xxm:xxs:xxxxs/")
                - len(self.unit)
                - 5
            )
        else:
            self.max_bars = max_bars

    def bar_str(self):
        percentage = min(self.num_complete / self.total, 1)

        num_bars = int(percentage * self.max_bars)
        markers = "█" * num_bars + self.none_char * (self.max_bars - num_bars)
        if self.num_complete > self.total:
            # replace last marker with >
            markers = markers[:-2] + " >"
        percentage = int(percentage * 100)

        time_passed = int(time.time() - self.start_time)
        # convert time passed to minutes:seconds
        time_passed_str = f"{time_passed // 60}m:{time_passed % 60}s"

        rate = time_passed / self.num_complete if self.num_complete > 0 else 0
        return f"{self.desc}|{markers}| {self.num_complete}/{self.total} ({percentage}%)|{time_passed_str}:{rate:.2f}s/{self.unit}"

    def print_bar(self):
        print(f"\r{self.bar_str()}", end="", flush=True)

    def increment(self):
        self.num_complete += 1

    # to string
    def __str__(self):
        return self.bar_str()


if __name__ == "__main__":
    import time

    mx = 30
    bar = manual_bar_printer(10, desc="Processing", unit="bits", auto_size=True)
    for i in range(mx):
        time.sleep(0.1)
        bar.increment()
        bar.print_bar()
