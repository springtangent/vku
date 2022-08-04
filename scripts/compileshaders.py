from . import config
from . import util


TARGET_PATH = "build\\resources"


def main():
	util.compile_shaders(config.RESOURCE_PATH)


if __name__ == "__main__":
	main()