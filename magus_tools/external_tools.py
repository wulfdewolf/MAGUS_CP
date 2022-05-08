"""
Created on Apr 14, 2020

@author: Vlad
"""

import subprocess
import os
import random
import shutil
import numpy
from magus_configuration import Configs
from magus_tasks.task import Task


def runCommand(**kwargs):
    command = kwargs["command"]
    Configs.log("Running an external tool, command: {}".format(command))
    runner = subprocess.run(
        command,
        shell=True,
        cwd=kwargs["workingDir"],
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        runner.check_returncode()
    except:
        Configs.error("Command encountered error: {}".format(command))
        Configs.error("Exit code: {}".format(runner.returncode))
        Configs.error("Output: {}".format(runner.stdout))
        raise
    for srcPath, destPath in kwargs.get("fileCopyMap", {}).items():
        shutil.move(srcPath, destPath)


def runClustalOmegaGuideTree(fastaPath, workingDir, outputPath, threads=1):
    tempPath = os.path.join(
        os.path.dirname(outputPath), "temp_{}".format(os.path.basename(outputPath))
    )
    args = [Configs.clustalPath]
    args.extend(
        [
            "-i",
            fastaPath,
            "--max-hmm-iterations=-1",
            "--guidetree-out={}".format(tempPath),
        ]
    )
    args.extend(["--threads={}".format(threads)])
    taskArgs = {
        "command": subprocess.list2cmdline(args),
        "fileCopyMap": {tempPath: outputPath},
        "workingDir": workingDir,
    }
    return Task(taskType="runCommand", outputFile=outputPath, taskArgs=taskArgs)


def generateMafftFilePathMap(inputPaths, outputDir):
    mafftMap = {
        inputPath: os.path.join(
            outputDir, "mafft_{}".format(os.path.basename(inputPath))
        )
        for inputPath in inputPaths
    }
    return mafftMap


def buildMafftAlignments(inputOutputPathMap):
    tasks = [
        buildMafftAlignment(inputPath, outputPath)
        for inputPath, outputPath in inputOutputPathMap.items()
    ]
    return tasks


def buildMafftAlignment(inputPath, outputPath, subtablePath=None):
    return runMafft(
        inputPath, subtablePath, Configs.workingDir, outputPath, Configs.numCores
    )


def runMafft(fastaPath, subtablePath, workingDir, outputPath, threads=1):
    tempPath = os.path.join(
        os.path.dirname(outputPath), "temp_{}".format(os.path.basename(outputPath))
    )
    args = [
        Configs.mafftPath,
        "--localpair",
        "--maxiterate",
        "1000",
        "--ep",
        "0.123",
        "--quiet",
        "--thread",
        str(threads),
        "--anysymbol",
    ]
    if subtablePath is not None:
        args.extend(["--merge", subtablePath])
    args.extend([fastaPath, ">", tempPath])
    taskArgs = {
        "command": subprocess.list2cmdline(args),
        "fileCopyMap": {tempPath: outputPath},
        "workingDir": workingDir,
    }
    return Task(taskType="runCommand", outputFile=outputPath, taskArgs=taskArgs)


def runMafftGuideTree(fastaPath, workingDir, outputPath, threads=1):
    tempPath = os.path.join(
        os.path.dirname(outputPath), "temp_{}".format(os.path.basename(outputPath))
    )
    treeFile = os.path.join(
        os.path.dirname(fastaPath), "{}.tree".format(os.path.basename(fastaPath))
    )
    args = [
        Configs.mafftPath,
        "--retree",
        "0",
        "--treeout",
        "--parttree",
        "--quiet",
        "--thread",
        str(threads),
        "--anysymbol",
    ]
    args.extend(["--partsize", "1000"])
    args.extend([fastaPath, ">", tempPath])
    taskArgs = {
        "command": subprocess.list2cmdline(args),
        "fileCopyMap": {treeFile: outputPath},
        "workingDir": workingDir,
    }
    return Task(taskType="runCommand", outputFile=outputPath, taskArgs=taskArgs)


def runMcl(matrixPath, inflation, workingDir, outputPath):
    tempPath = os.path.join(
        os.path.dirname(outputPath), "temp_{}".format(os.path.basename(outputPath))
    )
    args = [Configs.mclPath, matrixPath, "--abc", "-o", tempPath]
    if inflation is not None:
        args.extend(["-I", str(inflation)])
    taskArgs = {
        "command": subprocess.list2cmdline(args),
        "fileCopyMap": {tempPath: outputPath},
        "workingDir": workingDir,
    }
    return Task(taskType="runCommand", outputFile=outputPath, taskArgs=taskArgs)


def runMlrMcl(matrixPath, granularity, balance, inflation, workingDir, outputPath):
    tempPath = os.path.join(
        os.path.dirname(outputPath), "temp_{}".format(os.path.basename(outputPath))
    )
    args = [Configs.mlrmclPath, matrixPath, "-o", tempPath]
    if granularity is not None:
        args.extend(["-c", str(granularity)])
    if balance is not None:
        args.extend(["-b", str(balance)])
    if inflation is not None:
        args.extend(["-i", str(inflation)])
    taskArgs = {
        "command": subprocess.list2cmdline(args),
        "fileCopyMap": {tempPath: outputPath},
        "workingDir": workingDir,
    }
    return Task(taskType="runCommand", outputFile=outputPath, taskArgs=taskArgs)


def runFastTree(fastaFilePath, workingDir, outputPath, mode="normal", intree=None):
    tempPath = os.path.join(
        os.path.dirname(outputPath), "temp_{}".format(os.path.basename(outputPath))
    )

    args = [Configs.fasttreePath]
    if Configs.inferDataType(fastaFilePath) == "protein":
        args.extend(["-lg"])
    else:
        args.extend(["-nt", "-gtr"])

    if intree is not None:
        args.extend(["-intree", intree])

    if mode == "fast":
        args.extend(["-fastest", "-nosupport"])
    elif mode == "faster":
        args.extend(["-fastest", "-nosupport", "-mlnni", "4"])
    elif mode == "noml":
        args.extend(["-fastest", "-nosupport", "-noml"])

    args.extend([fastaFilePath, ">", tempPath])
    taskArgs = {
        "command": subprocess.list2cmdline(args),
        "fileCopyMap": {tempPath: outputPath},
        "workingDir": workingDir,
    }
    return Task(taskType="runCommand", outputFile=outputPath, taskArgs=taskArgs)


def runRaxmlNg(fastaFilePath, workingDir, outputPath, threads=8):
    # raxml-ng --msa prim.phy --model GTR+G --prefix T4 --threads 2 --seed 2 --tree pars{25},rand{25}
    baseName = os.path.basename(outputPath).replace(".", "")
    raxmlFile = os.path.join(workingDir, "{}.raxml.bestTree".format(baseName))
    seed = random.randint(1, 1000000)
    args = [
        Configs.raxmlPath,
        "--msa",
        fastaFilePath,
        "--prefix",
        baseName,
        "--threads",
        str(threads),
        "--seed",
        str(seed),
    ]

    if Configs.inferDataType(fastaFilePath) == "protein":
        args.extend(["--model", "LG+G"])
    else:
        args.extend(["--model", "GTR+G"])

    args.extend(["--tree", "pars{{{}}}".format(1)])
    taskArgs = {
        "command": subprocess.list2cmdline(args),
        "fileCopyMap": {raxmlFile: outputPath},
        "workingDir": workingDir,
    }
    return Task(taskType="runCommand", outputFile=outputPath, taskArgs=taskArgs)


def runHmmBuild(alignmentPath, workingDir, outputPath):
    tempPath = os.path.join(
        os.path.dirname(outputPath), "temp_{}".format(os.path.basename(outputPath))
    )
    args = [Configs.hmmbuildPath, "--ere", "0.59", "--cpu", "1"]
    args.extend(["--symfrac", "0.0", "--informat", "afa", tempPath, alignmentPath])
    taskArgs = {
        "command": subprocess.list2cmdline(args),
        "fileCopyMap": {tempPath: outputPath},
        "workingDir": workingDir,
    }
    return Task(taskType="runCommand", outputFile=outputPath, taskArgs=taskArgs)


def runHmmAlign(hmmModelPath, fragPath, workingDir, outputPath):
    tempPath = os.path.join(
        os.path.dirname(outputPath), "temp_{}".format(os.path.basename(outputPath))
    )
    args = [Configs.hmmalignPath, "-o", tempPath]
    args.extend([hmmModelPath, fragPath])
    taskArgs = {
        "command": subprocess.list2cmdline(args),
        "fileCopyMap": {tempPath: outputPath},
        "workingDir": workingDir,
    }
    return Task(taskType="runCommand", outputFile=outputPath, taskArgs=taskArgs)


def runHmmSearch(hmmModelPath, fragPath, workingDir, outputPath):
    tempPath = os.path.join(
        os.path.dirname(outputPath), "temp_{}".format(os.path.basename(outputPath))
    )
    args = [
        Configs.hmmsearchPath,
        "--noali",
        "--cpu",
        "1",
        "-o",
        tempPath,
        "-E",
        "99999999",
        "--max",
    ]
    args.extend([hmmModelPath, fragPath])
    taskArgs = {
        "command": subprocess.list2cmdline(args),
        "fileCopyMap": {tempPath: outputPath},
        "workingDir": workingDir,
    }
    return Task(taskType="runCommand", outputFile=outputPath, taskArgs=taskArgs)


def runMinizincTrace(graph, upper_bound):

    # Input with zeroes
    input = [0] * graph.matrixSize
    for cluster_id, cluster in enumerate(graph.clusters):
        for node in cluster:
            input[node] = cluster_id + 1
    subalignment_start = graph.subsetMatrixIdx + [len(graph.matrix)]

    # Get non-zero version
    input_non_zero = []
    subalignment_count = 1
    subalignment_start_non_zero = [subalignment_count]
    for subalignment_idx in range(len(subalignment_start) - 1):
        for input_idx in range(
            subalignment_start[subalignment_idx],
            subalignment_start[subalignment_idx + 1],
        ):
            if input[input_idx] != 0:
                input_non_zero.append(input[input_idx])
                subalignment_count += 1
        subalignment_start_non_zero.append(subalignment_count)
    subalignment_start_non_zero += [len(input_non_zero) + 1]

    n_alignments = len(graph.subalignmentLengths)
    input_length = len(input_non_zero)

    # Write instance
    with open(os.path.join(graph.workingDir, "instance.mzn"), "w") as instance_file:
        instance_file.write(
            f"nr_alignments = {n_alignments};\n"
            + f"input_length = {input_length};\n"
            + f"max_id = {upper_bound};\n"
            + "input = ["
            + ",".join([str(val) for val in input_non_zero])
            + "];\n"
            + "subalignment_start = ["
            + ",".join([str(val) for val in subalignment_start_non_zero])
            + "];\n"
        )

    # Run minizinc
    tempPath = os.path.join(
        os.path.dirname(graph.tracePath),
        "temp_{}".format(os.path.basename(graph.tracePath)),
    )
    args = [
        Configs.minizincPath,
        "-o",
        tempPath,
        "--model",
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "minizinc/tracing.mzn"
        ),
        "--solver",
        "gecode",
        os.path.join(graph.workingDir, "instance.mzn"),
    ]
    taskArgs = {
        "command": subprocess.list2cmdline(args),
        "fileCopyMap": {tempPath: graph.tracePath},
        "workingDir": graph.workingDir,
    }
    return (
        Task(taskType="runCommand", outputFile=graph.tracePath, taskArgs=taskArgs),
        input,  # Return input with zeroes to allow decoding
    )
