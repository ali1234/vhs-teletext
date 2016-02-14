class AllClass(object):
    """An object which contains everything."""
    def __contains__(self, other):
        return True

All = AllClass()
