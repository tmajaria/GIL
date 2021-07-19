version 1.0
workflow bdc_nifti_imaging {
    input {
        Int? memory
        Int? boot_disk
        Int? disk
    }
    call test_gpu {
        input:
            memory = memory,
            boot_disk = boot_disk,
            disk = disk
    }
    meta {
        author: "https://intellipaat.com/community/33459/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell"
    }
}

task test_gpu {
    input {
        Int? memory
        Int? boot_disk
        Int? disk
    }
    command <<<
        python3 <<CODE
        import tensorflow as tf 
        if tf.test.gpu_device_name(): 
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")
        CODE
    >>>
    runtime {
        docker: "tmajarian/helxplatform_gil:latest"
        memory: select_first([memory,"16"]) + " GB"
        disks: "local-disk " + select_first([disk,"128"]) + " SSD"
        bootDiskSizeGb: select_first([boot_disk,"50"])
        cpu: 4
        gpuCount: 2
        gpuType: "nvidia-tesla-k80"
        nvidiaDriverVersion: "450.51.06"
        zones: ["us-central1-c"]
    }
}
