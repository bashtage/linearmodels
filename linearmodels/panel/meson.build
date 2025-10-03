if not get_option('no-binary')
    py.extension_module(
        '_utility',
        '_utility.pyx',
        install: true,
        include_directories: [inc_np],
        subdir: 'linearmodels/panel',
        cython_args: cython_args,
        c_args: cython_c_args,
    )
endif
