import weakref

class Static(object):
    def __init__(self):
        self.instances = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        try:
            return self.instances[instance]
        except KeyError:
            raise AttributeError(str(type(self)) + " has not been set on " + str(owner))



class PageNumber(Static):
    """Page number. Found in header packets and row 27 fastext links."""

    def __set__(self, instance, value):
        if value < 0 or value > 0xff:
            raise ValueError("Page numbers must be between 0 and 0xff.")
        self.instances[instance] = int(value)



class MagazineNumber(Static):
    """Magazine number from 0-7."""

    def __set__(self, instance, value):
        if value < 0 or value > 7:
            raise ValueError("Magazine numbers must be between 0 and 7.")
        self.instances[instance] = int(value)



class SubpageNumber(Static):
    """Subpage number from 0-0x3f7f."""

    def __set__(self, instance, value):
        if value < 0 or value > 0x3f7f:
            raise ValueError("Subpage numbers must be between 0 and 0x3f7f.")
        if value & 0x80:
            raise ValueError("Bit 8 of subpage numbers must be 0.")
        self.instances[instance] = int(value)



class RowNumber(Static):
    """Row number from 0-31."""

    def __set__(self, instance, value):
        if value < 0 or value > 31:
            raise ValueError("Row numbers must be between 0 and 31.")
        self.instances[instance] = int(value)



class ControlBits(Static):
    """Row number from 0-31."""

    def __set__(self, instance, value):
        if value < 0 or value > 2047:
            raise ValueError("Control bits must be between 0 and 2047.")
        self.instances[instance] = int(value)
