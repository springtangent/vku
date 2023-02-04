import os
import os.path
import pathlib
import shutil

from . import config


def find_files_with_extensions(path, exts):
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            if file.endswith(exts):
                yield os.path.relpath(dirpath, path), file


def find_shaders(path):
    yield from find_files_with_extensions(path, config.SHADER_SOURCE_EXTENSIONS)


def find_compiled_shaders(path):
    yield from find_files_with_extensions(path, config.SHADER_OBJECT_EXTENSIONS)


def get_output_filename(dname, fname):
    return fname + '.spv'


def compile_shader(shader_path, input_filename, output_filename):
    print('compiling shader:', shader_path, output_filename, input_filename)
    output_file_path = os.path.join(shader_path, output_filename)
    input_file_path = os.path.join(shader_path, input_filename)
    cmd = f"{config.VULKAN_GLSLC_EXECUTABLE} {input_file_path} -o {output_file_path}"
    print(cmd)
    os.system(cmd)


def compile_shaders(path):
    print(f"compiling shaders in \"{path}\"")
    print('--')
    for input_dirname, input_filename in find_files_with_extensions(path, config.SHADER_SOURCE_EXTENSIONS):
        output_filename = get_output_filename(input_dirname, input_filename)
        print(f"input_dirname: {input_dirname} input filename: {input_filename}, output filename: {output_filename}")
        compile_shader(os.path.join(path, input_dirname), input_filename, output_filename)
        print('-')
        print()


def copy_files_with_extensions(src_path, dst_path, extensions):
    print("copy_files_with_extensions:", src_path, dst_path, extensions)
    for dirname, fname in find_files_with_extensions(src_path, extensions):
        print(dirname, fname)
        src = os.path.join(src_path, dirname, fname)
        dst = os.path.join(dst_path, dirname, fname)
        dst_dir = os.path.join(dst_path, dirname)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        print("copying file:", src, dst)
        shutil.copy(src, dst)

