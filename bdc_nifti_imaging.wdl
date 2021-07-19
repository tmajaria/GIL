version 1.0
workflow bdc_nifti_imaging {
    input {
        Array[File] image_files
        Array[String] image_labels
        Float? test_ratio
        Int? epochs
        Int? batch_size
        Int? memory
        Int? boot_disk
        Int? disk
    }
    call train {
        input:
            image_files=image_files,
            image_labels=image_labels,
            test_ratio = test_ratio,
            epochs = epochs,
            batch_size = batch_size,
    }
    output {
        File model = train.model
        File model_loss = train.model_loss
        File image_sizes = train.image_sizes
    }
    meta {
        description: "This workflow demonstrates model training for image classification as a POC for Biodata Catalyst image analysis."
        tags: "Image analysis"
        author: "M. Baumann"
    }
    # Parameter metadata
    parameter_meta {
        image_files: "Array of image filepaths"
        image_labels: "Array of image labels"
        test_ratio: "Percentage for testing data. Default is 0.3 (30%)"
        epochs: "Number of epochs. Default is 15"
        batch_size: "Training batch size. Default is 8"
        memory: "runtime parameter - amount of memory to allocate in GB. Default is: 16"
        boot_disk: "runtime parameter - amount of boot disk space to allocate in GB. Default is: 50"
        disk: "runtime parameter - amount of disk space to allocate in GB. Default is: 128"
    }
}

task train {
    input {
        Array[File] image_files
        Array[String] image_labels
        String filename_column_name = "image_filename"
        String label_column_name = "image_label"
        String csv_filename = "./image_data_csv"
        Float? test_ratio
        Int? epochs
        Int? batch_size
        Int? memory
        Int? boot_disk
        Int? disk
    }
    command {
        set -ux
        # Debug: Output the lists of image_files
        echo image_files: "${sep='", "' image_files}"
        echo image_labels: "${sep='", "' image_labels}"

        python3 <<CODE
        import re
        files = ["~{sep='", "' image_files}"]
        labels = ["~{sep='", "' image_labels}"]
        with open("~{csv_filename}", "w") as csv_file:
            csv_file.write("~{filename_column_name},~{label_column_name}\n")
            for i in range(len(files)):
                label = re.sub("[^0-9]","", labels[i]) # Hack: Tensorflow requires numeric labels
                csv_file.write(files[i] + "," + label + "\n")
        CODE

        # Debug: Output the contents of the CSV file
        cat ~{csv_filename}

        # Debug
        nvidia-smi
        /usr/local/cuda/bin/nvcc --version

        python3 /opt/GIL/get_sizes.py --data_csv ~{csv_filename} --image_column ~{filename_column_name} 

        python3 /opt/GIL/train.py --auto_resize --data_csv ~{csv_filename} --image_column ~{filename_column_name} --label_column ~{label_column_name} ~{"--test_ratio " + test_ratio} ~{"--epochs " + epochs} ~{"--batch_size " + batch_size}
    }
    output {
        File model = "model.h5"
        File model_loss = "model_loss.csv"
        File image_sizes = "image_sizes.csv"
    }
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
