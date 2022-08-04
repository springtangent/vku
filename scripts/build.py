import os

def main():
	os.system('cmake -S . -B build')
	os.system('cmake --build build')

if __name__ == "__main__":
	main()