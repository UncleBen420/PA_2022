from VideoPreprocessorFromFolder import VideoPreprocessorFromFolder

input = 'test_video/Input'
output = 'test_video/Output'

vpff = VideoPreprocessorFromFolder()

#vpff.process("opticalFlowSimple", input, output)

#vpff.process("opticalFlowFarneback", input, output)

#vpff.process("pifpaf", input, output)

#vpff.process("pifpaf_opticalflow", input, output)

vpff.process("rescale", input, output)
