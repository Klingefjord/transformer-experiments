class Config:
    """A class to hold dynamic configuration parameters for a model."""

    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)

def bytes_to_unicode():
    """
    Taken from GPT-2 encoder.
    Map every byte 0-255 to a readable unicode character.
    """

    # the 188 integers that render fine in their original form and need no shifting
    base_chars = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    # all integers b in base_chars will simply map to chr(b) in the output dict
    chars = base_chars[:]

    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    n = 0
    for b in range(2**8):
        if b not in base_chars:
            base_chars.append(b)
            chars.append(2**8 + n)
            n += 1

    chars = [chr(n) for n in chars]
    return dict(zip(base_chars, chars))