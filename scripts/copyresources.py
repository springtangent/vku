from . import config
from . import util
import os.path

TARGET_PATH = os.path.join("build","resources")


def main():
	util.copy_files_with_extensions(config.RESOURCE_PATH, TARGET_PATH, config.SHADER_OBJECT_EXTENSIONS)


if __name__ == "__main__":
	main()