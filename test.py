import time
import unittest

class MyTestCase(unittest.TestCase):
    def test_time_difference(self):
        last_updated_time = time.time() - 15  # 假设上次更新时间为 5 秒前
        time_difference = abs(time.time() - last_updated_time)
        print(time_difference)
        self.assertTrue(time_difference >= 10)

if __name__ == '__main__':
    unittest.main()