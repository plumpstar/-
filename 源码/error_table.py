'''
定义错误表的数据
'''

empty_elem = 'n/a'

class error_table():
    def __init__(self, **kwargs):
        if 'keys' in kwargs:
            print('Initializing the error table with the following elements: ')
            print(kwargs['keys'])
            self.entries = dict.fromkeys(kwargs['keys'], [])
        else:
            self.entries = {}
        self.num_elems = 0

    def update_with_elem(self, new_dict):
        # 先更新公用的部分
        common_keys = set(new_dict.keys()) & set(self.entries.keys())
        for k in common_keys:
            self.entries[k].append(new_dict[k])

        # 第二次更新新字典中不存在的条目
        absent_keys = set(self.entries.keys()) - set(new_dict.keys())
        for k in absent_keys:
            self.entries[k].append(empty_elem)

        # 第三次计算条目中不存在的
        missing_keys = set(new_dict.keys()) - set(self.entries.keys())
        for k in missing_keys:
            self.entries[k] = [empty_elem]*self.num_elems + [new_dict[k]]

        self.num_elems += 1



