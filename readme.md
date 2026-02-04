# 更好的Python编程

## 案例分析

琪露诺的编程实战特训
1. Day 1: 顺序化编程，但是心怀作用域；超参数思想
   - detect.py: 按颜色分割并进行形态学处理实现红色物体检测
2. Day 2: 典型的管道式编程
   - dehaze.py: 何凯明的暗通道先验算法去雾，含改进
   - count.py: 阈值分割和分水岭算法实现的图像计数
3. Day 3: 多层次管线的协同配合
   - segment.py: 肺部病灶分割（含肺部分割、病灶分割两条管线）的简易算法
4. Day 4: 循环性和记忆性复杂结构的管线
   - track.py: 模板匹配+卡尔曼滤波的目标跟踪算法
   - yolotrack.py: 预训练YOLO+卡尔曼滤波的目标跟踪算法
5. Day 5: 模块插入的管线——监督学习的部署
   - dehazenet.py: DehazeNet论文复现：部署
6. Day 6: 参数优化的管线——监督学习的训练
   - dehazenet.py: DehazeNet论文复现：训练
8. Day 7: 美国大学生数学建模竞赛：ODE动态建模
9. 优化算法的本质：求解器、机械臂逆运动学、反向传播、路径规划
10. Day 8: 循环性和记忆性复杂结构的监督学习
   - Mediapipe-Hand(BlazeHand/BlazePalm)论文复现
11. Day 9: 非监督学习与生成模型
   - Stable Diffusion复现
12. Day 10: 强化学习
    - 强化学习训练PyTouhou
13. Day 11 to 20: 强化学习实战
    - 训练你的RoboCup机器人，调整机器人的策略，踢出最佳风采吧！


7. Day 7: 可视化与调试功能的探索
## 完整的面向对象结构，服务于输入输出

下面是一个完整的类的结构（不包括Python一些高级语法，仅满足基本输入输出需求）

```python
class SomeModule:

    '''
    这里的量属于整个类，由类的所有对象共享（我不推荐把它作为变量，这不如设计一个顶层控制结构体对象）
    - 常量
    - 超参数配置
    '''
    SHARED_HYPERPARAMETERS_THAT_IS_CONFIG = 6
    CONSTANT_AND_SO_ON = 3.14159

    '''
    静态函数，不需要变量辅助，直接通过argument和return交换数据
    - 无记忆的工具函数（图像处理管道、机器学习部分算子）
        - 工厂函数用于生成对象
        - 测试函数直接执行用于生成对象并且测试其行为（主函数main或者单元测试test）
    '''
    @staticmethod
    def utility(argument):
        return argument
    @classmethod
    def factory_or_main(cls, argument):
        return cls.utility(argument) + cls.other_utility(argument)

    '''
    回调函数，被动触发，argument和return格式受编程框架（HAL库、Qt库……）限制，只能用实例变量交换数据
    - 绑定其他对象，修改其他对象的parameter
    - 修改自身对象的signal
    （一般定义在较为顶层的类中，就像STM32的回调函数和main几乎是一个层次。因为一个中断/事件只能绑一个回调函数，这个回调函数处理各种各样的业务，权限必然会很高，会插手很多类的职责）
    '''
    def event_callback(self, event):
        if event:
            var1 = self.read_object.get_something()
            self.write_object.set_something(var1)
        return 'LIBRARY_OK'

    '''
    初始化需要让模块具有
    1. 绑定read对象句柄
    2. 绑定write句柄（回调函数的主要交换数据方法）
    3. 自己的parameter供其他对象修改
    4. 自己的signal供自己展示
    - （有时候某个内存地址属于各个对象共有写入权限，因此parameter和signal没那么明显的界限。不过到那个程度，应当放一个公共的对象，内部有多个parameter，并加互斥锁来进行访问控制。所以我们仍严格遵循parameter写入权不属于自己、signal写入权只属于自己的设定）
    - 内部私有的signal。当然，变量和函数都可以私有，这个不单独归为一类。
    - 具体代码（如打开文件、数据解析、加载环境）
    初始化时可以从argument传入来手动指定parameter。但是为了格式统一，argument只用来绑定对象，parameter在创建对象后再手动修改，或者在工厂函数中可以经过argument传入
    可开放set/get方法来修改/读取parameter。
    '''
    def __init__(self, r, w, w2):
        '''
        read     ->|--------|->write
        argument ->| MODULE |->return
        parameter->|--------|->signal
        '''

        '''object binding'''
        self.read_object = r
        self.write_object = w
        self.write_object2 = w2 or AnObject()
        '''依赖注入思想，实际上利用了Python短路求值，返回第一个“非False，非0，非None，非空”'''

        '''parameter and signal''' # do not use global variables
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.signal = False
        self._count = 0 # private signal

        some_widget.register(event_callback)

    '''
    自由度最高的行为业务逻辑，输入输出形式很自由（我们最常见的是这样的代码，但很容易因为它而忘了上面回调函数、静态函数那样受限制的代码所造成的代码规范）
    - 有记忆的工具函数，仍使用argument和return来交换数据。当然，可以用实例变量记录状态，例如图像处理的中间过程
    - 主动触发的执行函数（回调函数是被动触发的）（如Thread.run()）仍使用实例变量交换数据。因为这里的交换数据本质上就是对状态的记录。可以return self实现方法链调用
    '''
    def behavior_or_process(self, argument):
        self._count += 1
        if self._count > 100:
            self._count = 0
            self.signal = 1
        else:
            self.signal = 0
        return argument + 2
```

再次重申：

- parameter只让别人写，signal只让自己写
- 工具函数严格使用argument和return交换数据，执行函数严格使用parameter和signal（包括使用句柄）交换数据
- argument-return是无记忆的数据交换（管道），parameter和signal是有记忆的数据交换（邮箱）。因此可以有arg输入+signal输出（set方法）、param输入+return输出（get方法）

函数也可以作为argument传入，不要疑虑。回调函数就是这样的，只不过是被动触发，而这里是主动触发。一个函数也是一个类。只要是无记忆的类，都可以作为argument

```python
    def run_a_function(self, argument, func):
        for i in range(len(func)):
            func[i](argument)
```

这是为了以后的图形化编程思想做铺垫
- 这种结构很方便转换为静态计算图，但是静态计算图有个唯一的不足就是无法动态改变结构（尤其是自指方面），毕竟你不能让集成电路凭空创造出自己的一个新外设。

## 低耦合高内聚

虽然有这么多成分，但是：
一个函数可以是一个类，一个结构体也可以是一个类；
上面的对象看似是一整个类，其实也可以拆分。

- 单一职责原则：彼此之间信息交换少的函数或结构体，不要粘合在一起，不易于区分职责
- 高内聚原则：彼此之间信息交换多的函数或结构体，不要强行拆分，造成过多的对象句柄

```python
class SensorReader:
    """职责1: 读取传感器数据"""
    def read(self):
        return raw_data

class DataProcessor:
    """职责2: 处理数据"""
    def process(self, raw_data):
        return processed_data

class DataWriter:
    """职责3: 写入数据"""
    def write(self, data):
        save_to_file(data)

class SensorPipeline:
    """协调者：组合各个单一职责的类"""
    def __init__(self):
        self.reader = SensorReader()
        self.processor = DataProcessor()
        self.writer = DataWriter()
    
    def run(self):
        raw = self.reader.read()
        processed = self.processor.process(raw)
        self.writer.write(processed)
```

## 控制器的上下级关系，继承 VS 包含（组合）

类与类之间存在上下级关系，上级对象可以通过句柄访问下级对象，作为控制器，做到统一调度，而下级对象只需要管好自己的事

```python
# 继承 vs 组合（包含）的实用示例
class BaseProcessor:
    def process(self, data):
        raise NotImplementedError

# 继承方式（is-a关系）
class SpecializedProcessor(BaseProcessor):
    def process(self, data):
        return data * 2

# 组合方式（has-a关系）
class FlexibleProcessor:
    def __init__(self, strategy):
        self.strategy = strategy  # 策略模式
    
    def process(self, data):
        return self.strategy.process(data)

# 使用
strategy = SpecializedProcessor()
processor = FlexibleProcessor(strategy)  # 更灵活，可运行时更换策略
```

我曾经喜欢用继承实现，不过用继承和用包含其实都可以，用继承可以少些一些访问链（a.b.c），用包含不用担忧命名的重复

## 命名小巧思

强烈建议增加类型提示（type hint）、文档字符串（docstring）

成员的访问权限可以通过命名区分

```python
class GoodDesign:
    # 公开接口
    def public_method(self): ...
    
    # 保护方法（子类可用）
    def _protected_method(self): ...
    
    # 私有实现
    def __private_method(self): ...
    
    # 属性访问器
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, new_value):
        self._value = new_value
```