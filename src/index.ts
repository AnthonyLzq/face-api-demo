import { readFileSync } from 'fs'
import tfjs from '@tensorflow/tfjs-node'
import faceApi from '@vladmandic/face-api'

const modelUrl = 'node_modules/@vladmandic/face-api/model'
const optionsSSDMobileNet = new faceApi.SsdMobilenetv1Options({
  minConfidence: 0.5,
  maxResults: 1
})

const getDescriptors = async (imageFile: string) => {
  const buffer = readFileSync(imageFile)
  const tensor = tfjs.node.decodeImage(buffer, 3)
  const faces = await faceApi
    .detectAllFaces(tensor, optionsSSDMobileNet)
    .withFaceLandmarks()
    .withFaceDescriptors()
  tfjs.dispose(tensor)

  return faces.map(face => face.descriptor)
}

const main = async (file1: string, file2: string) => {
  console.log('input images:', file1, file2)

  await tfjs.ready()
  await faceApi.nets.ssdMobilenetv1.loadFromDisk(modelUrl)
  await faceApi.nets.faceLandmark68Net.loadFromDisk(modelUrl)
  await faceApi.nets.faceRecognitionNet.loadFromDisk(modelUrl)

  const desc1 = await getDescriptors(file1)
  const desc2 = await getDescriptors(file2)
  const distance = faceApi.euclideanDistance(desc1[0], desc2[0]) // only compare first found face in each image

  console.log('distance between most prominant detected faces:', distance)
  console.log('similarity between most prominant detected faces:', 1 - distance)
}

main('./images/obama1.jpg', './images/obama2.jpg')
