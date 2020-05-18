
def __line__():
    """
    所在位置行号
    :return:
    """
    from sys import _getframe as __frame__
    return __frame__().f_back.f_lineno


def __fun__():
    """
    所在位置函数名
    :return:
    """
    from sys import _getframe as __frame__
    return __frame__().f_back.f_code.co_name


def __assert__(condition, error_text='', exit_code=1):
    from sys import _getframe as __frame__
    where_call = __frame__().f_back
    try:
        assert condition
    except AssertionError:
        print('\033[31m', end='')
        print('AssertionError')
        print(' * Filename: %s:' % where_call.f_code.co_filename)
        print(' * Module: %s' % where_call.f_code.co_name)
        print(' * Line: %s' % where_call.f_lineno)
        print(' * Description: %s' % error_text)
        print('\033[0m', end='')
        exit(exit_code)
