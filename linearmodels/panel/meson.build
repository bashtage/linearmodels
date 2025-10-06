if not get_option('no-binary')
    py.extension_module(
        '_utility',
        '_utility.pyx',
        install: true,
        include_directories: [inc_np],
        subdir: 'linearmodels/panel',
        c_args: cython_c_args,
        cython_args: cython_args,
    )
endif
