#ifndef NOMINMAX
#  define NOMINMAX
#endif

#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <dshow.h>
#include <ks.h>
#include <ksmedia.h>
#include <ksproxy.h>
#include <vidcap.h>

#include <pangolin/factory/factory_registry.h>
#include <pangolin/utils/timer.h>
#include <pangolin/video/drivers/uvc_mediafoundation.h>
#include <pangolin/video/iostream_operators.h>

namespace pangolin
{

static constexpr DWORD KS_CONTROL_NODE_ID_INVALID = ~0;

const GUID GUID_EXTENSION_UNIT_DESCRIPTOR_OV580{
        0x2ccb0bda, 0x6331, 0x4fdb, 0x85, 0x0e, 0x79, 0x05, 0x4d, 0xbd, 0x56, 0x71};

UvcMediaFoundationVideo::UvcMediaFoundationVideo(int vendorId, int productId, int deviceId, size_t width, size_t height, int fps)
    : size_bytes(0),
      mediaSource(nullptr),
      sourceReader(nullptr),
      baseFilter(nullptr),
      ksControl(nullptr),
      ksControlNodeId(KS_CONTROL_NODE_ID_INVALID)
{
    if(FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED)))
    {
        throw VideoException("CoInitializeEx failed");
    }

    if(FAILED(MFStartup(MF_VERSION)))
    {
        throw VideoException("MfStartup failed");
    }

    if(!FindDevice(vendorId, productId, deviceId))
    {
        throw VideoException("Unable to open UVC media source");
    }
    InitDevice(width, height, fps);

    device_properties[PANGO_HAS_TIMING_DATA] = true;
}

UvcMediaFoundationVideo::~UvcMediaFoundationVideo()
{
    DeinitDevice();
    HRESULT hr = MFShutdown();
    if(FAILED(hr))
    {
        pango_print_warn("MFShutdown failed with result %X", hr);
    }
    CoUninitialize();
}

void UvcMediaFoundationVideo::Start()
{

}

void UvcMediaFoundationVideo::Stop()
{
}

size_t UvcMediaFoundationVideo::SizeBytes() const
{
    return size_bytes;
}

const std::vector<pangolin::StreamInfo>& UvcMediaFoundationVideo::Streams() const
{
    return streams;
}

bool UvcMediaFoundationVideo::GrabNext(unsigned char* image, bool wait)
{
    IMFSample* sample = nullptr;
    DWORD streamIndex = 0;
    DWORD flags = 0;
    LONGLONG timeStamp;

    HRESULT hr = sourceReader->ReadSample(
            (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &streamIndex, &flags, &timeStamp, &sample);
    if(SUCCEEDED(hr))
    {
        if((flags & MF_SOURCE_READERF_ENDOFSTREAM) != 0)
        {
            return false;
        }

        if(!sample)
        {
            return false;
        }
    }

    IMFMediaBuffer* mediaBuffer = nullptr;
    if(SUCCEEDED(hr))
    {
        hr = sample->ConvertToContiguousBuffer(&mediaBuffer);
    }
    if(SUCCEEDED(hr))
    {
        // Use the 2D buffer interface if it's available
        IMF2DBuffer* mediaBuffer2d = nullptr;
        hr = mediaBuffer->QueryInterface(&mediaBuffer2d);
        if(SUCCEEDED(hr))
        {
            hr = mediaBuffer2d->ContiguousCopyTo(image, (DWORD)size_bytes);
            mediaBuffer2d->Release();
            mediaBuffer2d = nullptr;
        }
        else
        {
            // No 2D buffer is available
            byte* buffer;
            DWORD bufferSize = 0;
            hr = mediaBuffer->Lock(&buffer, nullptr, &bufferSize);
            if(SUCCEEDED(hr))
            {
                size_t copySize = std::min((size_t)bufferSize, size_bytes);
                memcpy(image, buffer, copySize);
                mediaBuffer->Unlock();
            }
        }
    }

    if(SUCCEEDED(hr))
    {
        using namespace std::chrono;
        frame_properties[PANGO_HOST_RECEPTION_TIME_US] = picojson::value(pangolin::Time_us(pangolin::TimeNow()));
    }

    if(mediaBuffer)
    {
        mediaBuffer->Release();
        mediaBuffer = nullptr;
    }

    if(sample)
    {
        sample->Release();
        sample = nullptr;
    }

    return SUCCEEDED(hr);
}

bool UvcMediaFoundationVideo::GrabNewest(unsigned char* image, bool wait)
{
    return GrabNext(image, wait);
}

int UvcMediaFoundationVideo::IoCtrl(uint8_t unit, uint8_t ctrl, unsigned char* data, int len, UvcRequestCode req_code)
{
    if(!ksControl || ksControlNodeId == KS_CONTROL_NODE_ID_INVALID)
    {
        return -1;
    }

    HRESULT hr;
    KSP_NODE s = {};
    ULONG ulBytesReturned;

    s.Property.Set = GUID_EXTENSION_UNIT_DESCRIPTOR_OV580;
    s.Property.Id = 2;
    s.NodeId = ksControlNodeId;

    s.Property.Flags = KSPROPERTY_TYPE_TOPOLOGY;
    if(req_code == UVC_GET_CUR)
    {
        s.Property.Flags |= KSPROPERTY_TYPE_GET;
    }
    else if(req_code == UVC_SET_CUR)
    {
        s.Property.Flags |= KSPROPERTY_TYPE_SET;
    }
    hr = ksControl->KsProperty((PKSPROPERTY)&s, sizeof(s), &data[0], len, &ulBytesReturned);
    if(FAILED(hr))
    {
        pango_print_error("KsProperty failed on UVC device");
        return -1;
    }

    return 0;
}

bool UvcMediaFoundationVideo::GetExposure(int& exp_us)
{
    pango_print_warn("GetExposure not implemented for UvcMediaFoundationVideo");
    return false;
}

bool UvcMediaFoundationVideo::SetExposure(int exp_us)
{
    pango_print_warn("SetExposure not implemented for UvcMediaFoundationVideo");
    return false;
}

bool UvcMediaFoundationVideo::GetGain(float& gain)
{
    pango_print_warn("GetGain not implemented for UvcMediaFoundationVideo");
    return false;
}

bool UvcMediaFoundationVideo::SetGain(float gain)
{
    pango_print_warn("SetGain not implemented for UvcMediaFoundationVideo");
    return false;
}


const picojson::value& UvcMediaFoundationVideo::DeviceProperties() const
{
    return device_properties;
}

const picojson::value& UvcMediaFoundationVideo::FrameProperties() const
{
    return frame_properties;
}

bool UvcMediaFoundationVideo::FindDevice(int vendorId, int productId, int deviceId)
{
    // Create attributes for finding UVC devices
    IMFAttributes* searchAttributes = nullptr;
    HRESULT hr = MFCreateAttributes(&searchAttributes, 1);
    if(FAILED(hr))
    {
        pango_print_error("Unable to create UVC device search attributes");
    }

    // Request video capture devices
    if(SUCCEEDED(hr))
    {
        hr = searchAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                       MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
        if(FAILED(hr))
        {
            pango_print_error("Unable to set UVC source type attribute");
        }
    }

    // Enumerate the devices
    IMFActivate** devices = nullptr;
    UINT32 deviceCount = 0;
    if(SUCCEEDED(hr))
    {
        hr = MFEnumDeviceSources(searchAttributes, &devices, &deviceCount);
        if(FAILED(hr))
        {
            pango_print_error("Unable to enumerate UVC device sources");
        }
    }

    std::wstring symLink;
    bool activatedDevice = false;
    if(SUCCEEDED(hr))
    {
        for(UINT32 i = 0; i < deviceCount; ++i)
        {
            // Get this device's sym link
            WCHAR* symLinkWCStr = nullptr;
            UINT32 symLinkLength = 0;
            hr = devices[i]->GetAllocatedString(
                    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &symLinkWCStr, &symLinkLength);
            if(FAILED(hr))
            {
                hr = S_OK;
                continue;
            }

            std::wstring checkSymLink(symLinkWCStr);

            // Check if this device matches the requested vendor ID and product ID
            if(!DeviceMatches(checkSymLink, vendorId, productId))
            {
                continue;
            }

            if(deviceId == i)
            {
                hr = devices[i]->ActivateObject(IID_PPV_ARGS(&mediaSource));
                activatedDevice = SUCCEEDED(hr);
                if(activatedDevice)
                {
                    symLink = std::move(checkSymLink);
                }
                break;
            }
        }

        if(!activatedDevice)
        {
            pango_print_error("Unable to activate UVC device source");
            hr = E_FAIL;
        }
    }

    for(UINT32 i = 0; i < deviceCount; ++i)
    {
        devices[i]->Release();
        devices[i] = nullptr;
    }
    devices = nullptr;

    CoTaskMemFree(devices);

    if(searchAttributes != nullptr)
    {
        searchAttributes->Release();
        searchAttributes = nullptr;
    }

    // Find the DirectShow device
    ICreateDevEnum* dshowDevices = nullptr;
    if(SUCCEEDED(hr))
    {
        hr = CoCreateInstance(CLSID_SystemDeviceEnum, 0, CLSCTX_INPROC, IID_ICreateDevEnum, (void**)&dshowDevices);
    }

    IEnumMoniker* videoInputEnumerator = nullptr;
    if(SUCCEEDED(hr))
    {
        hr = dshowDevices->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &videoInputEnumerator, 0);
    }

    if(SUCCEEDED(hr))
    {
        IMoniker* moniker = nullptr;
        while((hr = videoInputEnumerator->Next(1, &moniker, 0)) == S_OK)
        {
            IPropertyBag* propertyBag = nullptr;
            hr = moniker->BindToStorage(0, 0, IID_IPropertyBag, (void**)&propertyBag);
            if(FAILED(hr))
            {
                moniker->Release();
                moniker = nullptr;
                continue;
            }

            VARIANT variantPath;
            VariantInit(&variantPath);
            hr = propertyBag->Read(L"DevicePath", &variantPath, nullptr);

            bool pathMatches;
            if(SUCCEEDED(hr) && variantPath.vt == VT_BSTR)
            {
                // Determine if this is the correct device by comparing its path against the source's symbolic link
                // This breaks the rules, but it seems to be the only way to make sure it's the correct device

                // DirectShow and MediaFoundation appear to each create their own symbolic link which contains a GUID
                // Ignore the GUID portion of the link, leaving just the path information
                size_t braceOffset = symLink.find(L'{');

                pathMatches = 0 == std::wcsncmp(symLink.c_str(), variantPath.bstrVal, braceOffset);
            }
            else
            {
                pathMatches = false;
            }

            VARIANT variantFriendlyName;
            VariantInit(&variantFriendlyName);
            hr = propertyBag->Read(L"FriendlyName", &variantFriendlyName, nullptr);

            if(!pathMatches)
            {
                moniker->Release();
                moniker = nullptr;
                continue;
            }

            // Found the correct video input
            break;
        }

        if(moniker)
        {
            hr = moniker->BindToObject(0, 0, IID_IBaseFilter, (void**)&baseFilter);
        }
        else
        {
            hr = E_FAIL;
        }

        IKsTopologyInfo* ksTopologyInfo = nullptr;
        if(SUCCEEDED(hr))
        {
            hr = baseFilter->QueryInterface(__uuidof(IKsTopologyInfo), (void**)&ksTopologyInfo);
        }

        DWORD numberOfNodes = 0;
        if(SUCCEEDED(hr))
        {
            hr = ksTopologyInfo->get_NumNodes(&numberOfNodes);
        }

        GUID nodeGuid;
        for(DWORD nodeIndex = 0; nodeIndex < numberOfNodes; ++nodeIndex)
        {
            if(FAILED(ksTopologyInfo->get_NodeType(nodeIndex, &nodeGuid)))
            {
                continue;
            }

            if(nodeGuid == KSNODETYPE_DEV_SPECIFIC)
            {
                // This is the extension node
                IKsNodeControl* pUnknown = nullptr;
                hr = ksTopologyInfo->CreateNodeInstance(nodeIndex, __uuidof(IUnknown), (void**)&pUnknown);

                if(SUCCEEDED(hr) && pUnknown != nullptr)
                {
                    hr = pUnknown->QueryInterface(__uuidof(IKsControl), (void**)&ksControl);
                }
                if(SUCCEEDED(hr))
                {
                    ksControlNodeId = nodeIndex;
                }

                if(pUnknown)
                {
                    pUnknown->Release();
                    pUnknown = nullptr;
                }
            }
        }

        if(ksTopologyInfo)
        {
            ksTopologyInfo->Release();
            ksTopologyInfo = nullptr;
        }

        if(moniker)
        {
            moniker->Release();
            moniker = nullptr;
        }
    }

    if(videoInputEnumerator)
    {
        videoInputEnumerator->Release();
    }

    if(dshowDevices)
    {
        dshowDevices->Release();
    }

    return SUCCEEDED(hr);
}

void UvcMediaFoundationVideo::InitDevice(size_t width, size_t height, int fps)
{
    HRESULT hr = MFCreateSourceReaderFromMediaSource(mediaSource, nullptr, &sourceReader);
    if(FAILED(hr))
    {
        pango_print_error("Unable to create source reader from UVC media source");
    }

    // Find the closest supported resolution
    UINT32 stride;
    PixelFormat pixelFormat;
    if(SUCCEEDED(hr))
    {
        IMFMediaType* bestMediaType = nullptr;
        int bestError = std::numeric_limits<int>::max();
        UINT32 bestWidth;
        UINT32 bestHeight;
        UINT32 bestStride;
        GUID bestGuid;

        for(DWORD i = 0;; ++i)
        {
            IMFMediaType* checkMediaType;
            hr = sourceReader->GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, i, &checkMediaType);
            if(FAILED(hr))
            {
                if(hr == MF_E_NO_MORE_TYPES)
                {
                    // Reached the end of the available media types
                    hr = S_OK;
                }
                else
                {
                    pango_print_error("Failed to get UVC native media type");
                }
                break;
            }

            UINT32 checkWidth;
            UINT32 checkHeight;
            if(FAILED(MFGetAttributeSize(checkMediaType, MF_MT_FRAME_SIZE, &checkWidth, &checkHeight)))
            {
                checkWidth = 0;
                checkHeight = 0;
            }

            int checkError = abs(int(checkWidth) - int(width)) + abs(int(checkHeight) - int(height));
            if(bestError > checkError)
            {
                // Release the previous best
                if(bestMediaType)
                {
                    bestMediaType->Release();
                }

                bestError = checkError;
                bestMediaType = checkMediaType;
                bestWidth = checkWidth;
                bestHeight = checkHeight;

                if(FAILED(checkMediaType->GetGUID(MF_MT_SUBTYPE, &bestGuid)))
                {
                    bestGuid = {0};
                }

                bestStride = MFGetAttributeUINT32(checkMediaType, MF_MT_DEFAULT_STRIDE, checkWidth);
            }
            else
            {
                checkMediaType->Release();
            }
        }
        if(bestMediaType)
        {
            if(SUCCEEDED(hr))
            {
                sourceReader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, bestMediaType);
                width = bestWidth;
                height = bestHeight;
                stride = bestStride;

                if(bestGuid == MFVideoFormat_YUY2)
                {
                    pixelFormat = PixelFormatFromString("GRAY8");
                }
                else
                {
                    pango_print_warn("Unexpected MFVideoFormat with FOURCC %c%c%c%c",
                                     (unsigned char)((bestGuid.Data1 >> 0) & 0xff),
                                     (unsigned char)((bestGuid.Data1 >> 8) & 0xff),
                                     (unsigned char)((bestGuid.Data1 >> 16) & 0xff),
                                     (unsigned char)((bestGuid.Data1 >> 24) & 0xff));
                }
            }

            bestMediaType->Release();
        }
        else
        {
            width = 0;
            height = 0;
        }
    }

    size_bytes = stride * height;

    streams.emplace_back(pixelFormat, width, height, stride);
}

void UvcMediaFoundationVideo::DeinitDevice()
{
    if(ksControl)
    {
        ksControl->Release();
        ksControl = nullptr;
    }

    ksControlNodeId = KS_CONTROL_NODE_ID_INVALID;

    if(baseFilter)
    {
        baseFilter->Release();
        baseFilter = nullptr;
    }

    if(sourceReader)
    {
        sourceReader->Release();
        sourceReader = nullptr;
    }

    if(mediaSource)
    {
        mediaSource->Shutdown();
        mediaSource->Release();
        mediaSource = nullptr;
    }

    streams.clear();
}

bool UvcMediaFoundationVideo::DeviceMatches(const std::wstring& symLink, int vendorId, int productId)
{
    // Example symlink:
    // \\?\usb#vid_05a9&pid_0581&mi_00#6&2ff327a4&2&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global
    //         ^^^^^^^^ ^^^^^^^^
    std::wstring symLinkString(symLink);
    if(vendorId != 0 && !SymLinkIDMatches(symLinkString, L"vid_", vendorId))
    {
        return false;
    }

    if(productId != 0 && !SymLinkIDMatches(symLinkString, L"pid_", productId))
    {
        return false;
    }

    return true;
}

bool UvcMediaFoundationVideo::SymLinkIDMatches(const std::wstring& symLink, const wchar_t* idStr, int id)
{
    // Find the ID prefix
    size_t idOffset = symLink.find(idStr);
    if(idOffset == std::wstring::npos)
    {
        // Unable to find the prefix
        return false;
    }

    // Parse the ID as a hexadecimal number
    return id == std::wcstol(&symLink[idOffset + std::wcslen(idStr)], nullptr, 16);
}

PANGOLIN_REGISTER_FACTORY(UvcMediaFoundationVideo)
{
    struct UvcVideoFactory final : public FactoryInterface<VideoInterface>
    {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override
        {
            int vendorId = 0;
            int productId = 0;
            std::istringstream(uri.Get<std::string>("vid", "0x0000")) >> std::hex >> vendorId;
            std::istringstream(uri.Get<std::string>("pid", "0x0000")) >> std::hex >> productId;
            const unsigned int deviceId = uri.Get<int>("num", 0);
            const ImageDim dim = uri.Get<ImageDim>("size", ImageDim(640, 480));
            const unsigned int fps = uri.Get<unsigned int>("fps", 0);  // 0 means unspecified
            return std::unique_ptr<VideoInterface>(new UvcMediaFoundationVideo(vendorId, productId, deviceId, dim.x, dim.y, fps));
        }
    };

    FactoryRegistry<VideoInterface>::I().RegisterFactory(std::make_shared<UvcVideoFactory>(), 10, "uvc");
}
}
