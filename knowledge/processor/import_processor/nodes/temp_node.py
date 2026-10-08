from knowledge.processor.import_processor.base import BaseNode, T


class TempNode(BaseNode):
    def process(self, state: T) -> T:
        """
        节点处理逻辑
        :param state:
        :return:
        """
        self.log_step("step1","准备开始执行xxx动作")


if __name__ == '__main__':
    pass



    # 方式一

    #   1. 实例当前对象
    # temp_node=TempNode()

    # 2. 调用process
    # temp_node.process()

    # 方式二(实体当做方法调用)
    # temp_node=TempNode()
    #
    # temp_node()
